// Native synchronous global batch. No server admission or shared prompt reuse.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <regex>
#include <set>
#include <map>
#include <cstdio>
using json=nlohmann::ordered_json;

struct PlacementAudit {
 std::set<std::string> cpu_expert_tensors;
 std::map<int,int> cpu_expert_layer_counts;
 std::map<int,std::string> layer_device;
};

static int64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static json read(const std::string &p){std::ifstream f(p);json j;f>>j;return j;}
static void write(const std::string &p,const json &j){std::ofstream f(p);f<<j.dump(2)<<"\n";}
static void require(bool b,const char *s){if(!b)throw std::runtime_error(s);}

static void audit_log(ggml_log_level level,const char *text,void *ud){
 (void)level;
 std::fputs(text,stderr);std::fflush(stderr);
 auto *a=static_cast<PlacementAudit *>(ud);
 static const std::regex re("tensor (blk\\.([0-9]+)\\.ffn_(up|down|gate|gate_up)_(ch|)exps[^ ]*) .*buffer type overridden to (CPU|CUDA_Host)(\\r?\\n|$)");
 static const std::regex re_layer("layer\\s+([0-9]+) assigned to device ([^,\\s]+)");
 std::cmatch m;
 if(std::regex_search(text,m,re)){
  std::string name=m[1].str();
  int layer=std::stoi(m[2].str());
  if(a->cpu_expert_tensors.insert(name).second)a->cpu_expert_layer_counts[layer]++;
 }
 if(std::regex_search(text,m,re_layer)){
  int layer=std::stoi(m[1].str());
  if(layer>=0 && layer<48)a->layer_device[layer]=m[2].str();
 }
}

int main(int argc,char **argv){try{
 require(argc==9||argc==10,"model warmup target out gpu_expert_layers repeats smoke cpu_threads [expert_placement]");
 std::string out=argv[4];
 int layers=std::stoi(argv[5]),repeats=std::stoi(argv[6]),threads=std::stoi(argv[8]);
 bool smoke=std::stoi(argv[7]);
 std::string placement_mode=argc==10?argv[9]:"legacy_tail";
 require(placement_mode=="balanced1","R8 requires one full expert layer per GPU");
 require(threads==16||threads==32||threads==64,"cpu_threads must be 16, 32, or 64");
 auto warm=read(argv[2])["requests"],target=read(argv[3])["requests"];
 int count=smoke?8:target.size(),len=smoke?32:target[0]["input_ids"].size(),n=smoke?2:64;
 require(count>0 && layers>=0 && layers<=48,"bad shape");

 PlacementAudit placement;
 llama_log_set(audit_log,&placement);
 ggml_backend_load_all();llama_backend_init();

 std::set<int> gpu_expert_layer_ids;
 if(placement_mode=="legacy_tail"){
  for(int i=48-layers;i<48;i++)gpu_expert_layer_ids.insert(i);
 }else if(placement_mode=="balanced3"){
  require(layers==12,"balanced3 requires exactly 12 GPU expert layers");
  const int ids[]={2,6,10,14,18,22,26,30,34,38,42,46};
  gpu_expert_layer_ids.insert(std::begin(ids),std::end(ids));
 }else{
  const int per=std::stoi(placement_mode.substr(8));
  require(layers==8*per,"R8 balanced placement layer count mismatch");
  for(int dev=0;dev<8;dev++)for(int j=0;j<per;j++){
   const int begin=dev*6,size=6;
   gpu_expert_layer_ids.insert(begin+(2*j+1)*size/(2*per));
  }
  require((int)gpu_expert_layer_ids.size()==layers,"balanced selection collision");
 }
 std::vector<std::string> patterns;
 for(int i=0;i<48;i++)if(!gpu_expert_layer_ids.count(i))patterns.push_back("blk\\."+std::to_string(i)+"\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
 std::vector<llama_model_tensor_buft_override> overrides;
 for(auto &s:patterns)overrides.push_back({s.c_str(),ggml_backend_cpu_buffer_type()});
 overrides.push_back({nullptr,nullptr});
 auto mp=llama_model_default_params();
 mp.n_gpu_layers=999;
 mp.split_mode=LLAMA_SPLIT_MODE_LAYER;
 mp.tensor_buft_overrides=overrides.data();
 auto model=llama_model_load_from_file(argv[1],mp);require(model,"model load");

 const int cpu_layers=48-layers;
 const int expected_cpu_tensors=cpu_layers*3;
 if((int)placement.cpu_expert_tensors.size()!=expected_cpu_tensors)throw std::runtime_error("CPU expert override count mismatch");
 for(int i=0;i<48;i++){
  int actual=placement.cpu_expert_layer_counts.count(i)?placement.cpu_expert_layer_counts.at(i):0;
  int expected=gpu_expert_layer_ids.count(i)?0:3;
  if(actual!=expected)throw std::runtime_error("CPU expert layer placement mismatch");
 }
 std::map<std::string,int> gpu_experts_by_device;
 json layer_device=json::object(),gpu_layer_ids=json::array(),gpu_by_device=json::object();
 for(auto &[layer,dev]:placement.layer_device)layer_device[std::to_string(layer)]=dev;
 for(int layer:gpu_expert_layer_ids){
  if(!placement.layer_device.count(layer))throw std::runtime_error("missing layer-to-device audit");
  gpu_experts_by_device[placement.layer_device.at(layer)]++;
  gpu_layer_ids.push_back(layer);
 }
 for(auto &[dev,cnt]:gpu_experts_by_device)gpu_by_device[dev]=cnt;
 if(placement_mode.rfind("balanced",0)==0){
  const int per=std::stoi(placement_mode.substr(8));
  require(placement.layer_device.size()==48,"balanced placement requires audited ownership for all transformer layers");
  require(gpu_experts_by_device.size()==8,"balanced placement must span eight CUDA devices");
  for(auto &[dev,cnt]:gpu_experts_by_device)require(cnt==per,"balanced placement violates equal per-device expert layers");
 }
 const int64_t expert_layer_bytes=int64_t(128)*9*1024*1024;
 json layer_counts=json::object();
 for(auto &[layer,cnt]:placement.cpu_expert_layer_counts)layer_counts[std::to_string(layer)]=cnt;
 write(out+"/placement_audit.json",{
  {"status","PASS"},
  {"verification","llama loader debug callback: exact CPU tensor overrides plus layer ownership"},
  {"expert_placement",placement_mode},
  {"gpu_expert_layer_ids",gpu_layer_ids},
  {"layer_device",layer_device},
  {"gpu_expert_layers_by_device",gpu_by_device},
  {"cpu_expert_layers",cpu_layers},
  {"gpu_expert_layers",layers},
  {"cpu_expert_tensor_count",(int)placement.cpu_expert_tensors.size()},
  {"gpu_expert_tensor_count",layers*3},
  {"expected_total_expert_tensor_count",144},
  {"cpu_expert_bytes",int64_t(cpu_layers)*expert_layer_bytes},
  {"gpu_expert_bytes",int64_t(layers)*expert_layer_bytes},
  {"per_cpu_layer_tensor_counts",layer_counts},
  {"op_offload",false}
 });

 auto cp=llama_context_default_params();
 cp.n_ctx=count*(len+n+8);
 cp.n_seq_max=count;
 cp.n_batch=2048;
 cp.n_ubatch=512;
 cp.n_threads=threads;
 cp.n_threads_batch=threads;
 cp.op_offload=false;
 cp.offload_kqv=true;
 auto ctx=llama_init_from_model(model,cp);require(ctx,"context load");
 const int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
 auto batch=llama_batch_init(cp.n_batch,0,1);
 auto add=[&](llama_token token,int pos,int seq,bool logits){int i=batch.n_tokens++;batch.token[i]=token;batch.pos[i]=pos;batch.n_seq_id[i]=1;batch.seq_id[i][0]=seq;batch.logits[i]=logits;};
 auto greedy=[&](int idx){float *v=llama_get_logits_ith(ctx,idx);require(v,"missing logits");int best=0;for(int i=0;i<vocab;i++){require(std::isfinite(v[i]),"nonfinite logits");if(v[i]>v[best])best=i;}return best;};

 for(int rep=0;rep<=repeats;rep++){
  auto rows=rep?target:warm;std::string phase=rep?"target":"warmup";
  write(out+"/phase.json",{{"phase",phase},{"repeat",rep},{"system","llama.cpp-sync"},{"cpu_threads",threads}});
  llama_synchronize(ctx);llama_memory_clear(llama_get_memory(ctx),true);
  std::vector<std::vector<int>> inputs(count),tokens(count);json ids=json::array();
  for(int r=0;r<count;r++){auto all=rows[r]["input_ids"].get<std::vector<int>>();inputs[r]=std::vector<int>(all.end()-len,all.end());ids.push_back(rows[r]["request_id"]);}
  std::vector<int> next(count,-1);std::vector<int64_t> stamps;int64_t start=ns();

  // Chunked prefill; no sequence enters decode before every prompt is complete.
  int total=count*len,prefill_calls=0;
  for(int off=0;off<total;){
   batch.n_tokens=0;std::vector<std::pair<int,int>> outputs;
   for(;off<total && batch.n_tokens<int(cp.n_batch);off++){
    int r=off/len,pos=off%len;bool last=pos==len-1;
    if(last)outputs.push_back({r,batch.n_tokens});
    add(inputs[r][pos],pos,r,last);
   }
   require(llama_decode(ctx,batch)==0,"prefill failed");llama_synchronize(ctx);prefill_calls++;
   for(auto [r,idx]:outputs)next[r]=greedy(idx);
  }
  for(int r=0;r<count;r++){require(next[r]>=0,"missing first token");tokens[r].push_back(next[r]);}
  stamps.push_back(ns());

  for(int step=1;step<n;step++){
   batch.n_tokens=0;
   for(int r=0;r<count;r++)add(next[r],len+step-1,r,true);
   require(llama_decode(ctx,batch)==0,"decode failed");llama_synchronize(ctx);
   for(int r=0;r<count;r++){next[r]=greedy(r);tokens[r].push_back(next[r]);}
   stamps.push_back(ns());
  }

  double ttft=(stamps.front()-start)/1e9,e2e=(stamps.back()-start)/1e9;
  std::vector<double> decode_steps;
  for(size_t i=1;i<stamps.size();i++)decode_steps.push_back((stamps[i]-stamps[i-1])/1e9);
  double tpot=(e2e-ttft)/(n-1);
  json result={
   {"status","PASS"},{"system","llama.cpp-sync"},{"repeat",rep},{"phase",phase},{"smoke",smoke},
   {"TTFT",ttft},{"TPOT",tpot},{"E2E",e2e},{"release_ns",start},{"token_ready_ns",stamps},
   {"decode_step_seconds",decode_steps},{"tokens",tokens},{"request_ids",ids},
   {"global_requests",count},{"input_tokens",len},{"output_tokens",n},
   {"prefill_tokens",total},{"prefill_decode_calls",prefill_calls},{"decode_calls",n-1},{"decode_tokens_per_call",count},
   {"n_batch",(int)cp.n_batch},{"n_ubatch",(int)cp.n_ubatch},
   {"cpu_threads",threads},{"cpu_batch_threads",threads},
   {"expert_placement",placement_mode},
   {"gpu_expert_layer_ids",gpu_layer_ids},
   {"gpu_expert_layers_by_device",gpu_by_device},
   {"gpu_expert_layers",layers},{"cpu_expert_layers",cpu_layers},
   {"expert_resident_bytes",int64_t(layers)*expert_layer_bytes},
   {"synchronous_batch",true},{"finite_logits",true},
   {"op_offload",false},{"offload_kqv",true},
   {"cache_start","empty KV; static weight partition retained"}
  };
  write(out+"/repeat"+std::to_string(rep)+".json",result);
  std::cout<<"repeat="<<rep<<" threads="<<threads<<" TTFT="<<ttft<<" TPOT="<<tpot<<" E2E="<<e2e<<std::endl;
 }
 llama_batch_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();
 llama_log_set(nullptr,nullptr);
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
