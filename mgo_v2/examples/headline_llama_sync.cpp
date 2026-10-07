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
using json=nlohmann::ordered_json;
static int64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static json read(const std::string &p){std::ifstream f(p);json j;f>>j;return j;}
static void write(const std::string &p,const json &j){std::ofstream f(p);f<<j.dump(2)<<"\n";}
static void require(bool b,const char *s){if(!b)throw std::runtime_error(s);}
int main(int argc,char **argv){try{
 require(argc==8,"model warmup target out gpu_expert_layers repeats smoke");
 std::string out=argv[4];int layers=std::stoi(argv[5]),repeats=std::stoi(argv[6]);bool smoke=std::stoi(argv[7]);
 auto warm=read(argv[2])["requests"],target=read(argv[3])["requests"];
 int count=smoke?4:target.size(),len=smoke?32:target[0]["input_ids"].size(),n=smoke?2:64;
 require(count>0 && layers>=0 && layers<=48,"bad shape");
 ggml_backend_load_all();llama_backend_init();
 std::vector<std::string> patterns;
 for(int i=0;i<48-layers;i++)patterns.push_back("blk\\."+std::to_string(i)+"\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
 std::vector<llama_model_tensor_buft_override> overrides;
 for(auto &s:patterns)overrides.push_back({s.c_str(),ggml_backend_cpu_buffer_type()});
 overrides.push_back({nullptr,nullptr});
 auto mp=llama_model_default_params();mp.n_gpu_layers=999;mp.split_mode=LLAMA_SPLIT_MODE_LAYER;mp.tensor_buft_overrides=overrides.data();
 auto model=llama_model_load_from_file(argv[1],mp);require(model,"model load");
 auto cp=llama_context_default_params();cp.n_ctx=count*(len+n+8);cp.n_seq_max=count;cp.n_batch=2048;cp.n_ubatch=512;cp.n_threads=32;cp.n_threads_batch=64;cp.op_offload=false;cp.offload_kqv=true;
 auto ctx=llama_init_from_model(model,cp);require(ctx,"context load");
 const int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
 auto batch=llama_batch_init(cp.n_batch,0,1);
 auto add=[&](llama_token token,int pos,int seq,bool logits){int i=batch.n_tokens++;batch.token[i]=token;batch.pos[i]=pos;batch.n_seq_id[i]=1;batch.seq_id[i][0]=seq;batch.logits[i]=logits;};
 auto greedy=[&](int idx){float *v=llama_get_logits_ith(ctx,idx);require(v,"missing logits");int best=0;for(int i=0;i<vocab;i++){require(std::isfinite(v[i]),"nonfinite logits");if(v[i]>v[best])best=i;}return best;};
 for(int rep=0;rep<=repeats;rep++){
  auto rows=rep?target:warm;std::string phase=rep?"target":"warmup";
  write(out+"/phase.json",{{"phase",phase},{"repeat",rep},{"system","llama.cpp-sync"}});
  llama_synchronize(ctx);llama_memory_clear(llama_get_memory(ctx),true);
  std::vector<std::vector<int>> inputs(count),tokens(count);json ids=json::array();
  for(int r=0;r<count;r++){auto all=rows[r]["input_ids"].get<std::vector<int>>();inputs[r]=std::vector<int>(all.end()-len,all.end());ids.push_back(rows[r]["request_id"]);}
  std::vector<int> next(count,-1);std::vector<int64_t> stamps;int64_t start=ns();
  // Chunked prefill, but no sequence advances into decode until all are ready.
  int total=count*len;
  for(int off=0;off<total;){batch.n_tokens=0;std::vector<std::pair<int,int>> outputs;
   for(;off<total && batch.n_tokens<int(cp.n_batch);off++){int r=off/len,pos=off%len;bool last=pos==len-1;if(last)outputs.push_back({r,batch.n_tokens});add(inputs[r][pos],pos,r,last);}
   require(llama_decode(ctx,batch)==0,"prefill failed");llama_synchronize(ctx);
   for(auto [r,idx]:outputs)next[r]=greedy(idx);
  }
  for(int r=0;r<count;r++){require(next[r]>=0,"missing first token");tokens[r].push_back(next[r]);}stamps.push_back(ns());
  for(int step=1;step<n;step++){batch.n_tokens=0;for(int r=0;r<count;r++)add(next[r],len+step-1,r,true);
   require(llama_decode(ctx,batch)==0,"decode failed");llama_synchronize(ctx);
   for(int r=0;r<count;r++){next[r]=greedy(r);tokens[r].push_back(next[r]);}stamps.push_back(ns());
  }
  double ttft=(stamps.front()-start)/1e9,e2e=(stamps.back()-start)/1e9;
  json result={{"status","PASS"},{"system","llama.cpp-sync"},{"repeat",rep},{"phase",phase},{"smoke",smoke},{"TTFT",ttft},{"TPOT",(e2e-ttft)/(n-1)},{"E2E",e2e},{"release_ns",start},{"token_ready_ns",stamps},{"tokens",tokens},{"request_ids",ids},{"global_requests",count},{"output_tokens",n},{"gpu_expert_layers",layers},{"expert_resident_bytes",int64_t(layers)*128*9*1024*1024},{"synchronous_batch",true},{"finite_logits",true},{"cache_start","empty KV; static weight partition retained"}};
  write(out+"/repeat"+std::to_string(rep)+".json",result);std::cout<<"repeat="<<rep<<" TTFT="<<ttft<<" TPOT="<<(e2e-ttft)/(n-1)<<" E2E="<<e2e<<std::endl;
 }
 llama_batch_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
