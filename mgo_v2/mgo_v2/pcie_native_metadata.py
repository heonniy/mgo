"""Opt-in current-routing pack kernel and C++ Gate history/histogram engine."""
import ctypes as C
import fcntl
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
import triton as tr
import triton.language as tl
from .live_metadata import LiveMetadata

SOURCE=Path(__file__).resolve().parents[1]/'scripts/native_pcie_metadata.cpp'
FLAGS=('-O3','-std=c++17','-shared','-fPIC','-ffp-contract=off')


@lru_cache(None)
def load_engine():
    digest=hashlib.sha256(SOURCE.read_bytes()+' '.join(FLAGS).encode()).hexdigest()[:16]
    root=Path('/data2/esjung/cache/native_pcie_metadata');root.mkdir(parents=True,exist_ok=True)
    path=root/f'metadata_{digest}.so'
    with (root/f'{digest}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not path.exists():
            temporary=path.with_suffix(f'.{os.getpid()}.tmp')
            try:
                subprocess.run(['g++',*FLAGS,str(SOURCE),'-o',str(temporary)],check=True,capture_output=True,text=True)
                temporary.replace(path)
            finally:temporary.unlink(missing_ok=True)
    lib=C.CDLL(str(path))
    lib.mgo_gate_update.argtypes=[C.c_void_p]*5+[C.c_int]*5
    lib.mgo_gate_update.restype=C.c_int
    lib.mgo_gate_scores.argtypes=[C.c_void_p]*3+[C.c_int]*2
    lib.mgo_route_histogram.argtypes=[C.c_void_p]*2+[C.c_int]*4
    lib.mgo_route_histogram.restype=C.c_int
    return lib


class WindowLength:
    def __init__(self,history,layer):self.history=history;self.layer=layer
    def __len__(self):return int(self.history.counts[self.layer])


class NativeGateHistory:
    def __init__(self,num_layers,num_experts,window=128):
        self.num_layers=num_layers;self.num_experts=num_experts;self.window=window
        self.ring=np.zeros((num_layers,window,num_experts),np.float64)
        self.sums=np.zeros((num_layers,num_experts),np.float64)
        self.scores=np.zeros((num_layers,num_experts),np.float32)
        self.heads=np.zeros(num_layers,np.int64);self.counts=np.zeros(num_layers,np.int64)
        self.rows=[WindowLength(self,layer) for layer in range(num_layers)]
        self.native=load_engine()
    def update(self,layer,probs):
        if probs is None:return
        if probs.ndim!=2 or probs.shape[1]!=self.num_experts:raise ValueError('probs must be [tokens, experts]')
        data=np.ascontiguousarray(probs,dtype=np.float64)
        status=self.native.mgo_gate_update(self.ring.ctypes.data,self.sums.ctypes.data,self.heads.ctypes.data,self.counts.ctypes.data,
              data.ctypes.data,self.num_layers,self.num_experts,self.window,layer,len(data))
        if status:raise ValueError('Invalid native Gate history geometry')
    def gates(self,layer):
        self.native.mgo_gate_scores(self.sums.ctypes.data,self.counts.ctypes.data,self.scores.ctypes.data,self.num_experts,layer)
        return self.scores[layer]
    def score(self,layer,expert):return float(self.gates(layer)[expert])


@tr.jit(do_not_specialize=['EVENT'])
def pack_current_metadata(Send,Selected,Probs,EVENT,BATCH:tl.constexpr,TOPK:tl.constexpr,
                          EXPERTS:tl.constexpr,TAIL:tl.constexpr,B:tl.constexpr=256):
    x=tl.program_id(0)*B+tl.arange(0,B)
    word=tl.where(x<8,1,tl.where(x<16,EVENT,BATCH)).to(tl.int64)
    byte=(word>>((x%8)*8))&255
    tl.store(Send+x,byte.to(tl.uint8),x<24)
    selected=tl.load(Selected+x,x<BATCH*TOPK,0).to(tl.uint8)
    tl.store(Send+24+x,selected,x<BATCH*TOPK)
    value=tl.load(Probs+(BATCH-TAIL)*EXPERTS+x,x<TAIL*EXPERTS,0)
    dest=(Send+24+BATCH*TOPK).to(tl.pointer_type(tl.float32))
    tl.store(dest+x,value,x<TAIL*EXPERTS)


class NativeLiveMetadata(LiveMetadata):
    def __init__(self,batch,history,topk=8,experts=128,layers=48):
        assert isinstance(history,NativeGateHistory)
        super().__init__(batch,history,topk,experts,layers)
        self.hist=np.empty((self.world,self.experts),np.int64)
        self.validate_wire=False;self.wire_checks=0
    def collect(self,event,selected,probs,frozen_gate=None):
        assert frozen_gate is None and selected.shape==(self.batch,self.topk) and probs.shape==(self.batch,self.experts)
        assert selected.dtype==torch.int64 and probs.dtype==torch.float32 and selected.is_contiguous() and probs.is_contiguous()
        pack_current_metadata[(tr.cdiv(max(24,self.batch*self.topk,self.tail*self.experts),256),)](
            self.send,selected,probs,event,self.batch,self.topk,self.experts,self.tail)
        if self.validate_wire and event<96:
            reference=torch.cat((torch.tensor([1,event,self.batch],device='cuda',dtype=torch.int64).view(torch.uint8),
                selected.to(torch.uint8).reshape(-1),probs[-self.tail:].contiguous().view(torch.uint8).reshape(-1)))
            assert torch.equal(self.send,reference),'Native metadata changed wire bytes'
            self.wire_checks+=1
        dist.all_gather_into_tensor(self.recv,self.send);self.host.copy_(self.recv);self.calls+=1
        raw=self.host.numpy().reshape(self.world,self.record_bytes)
        headers=raw[:,:24].copy().view(np.int64).reshape(self.world,3)
        assert np.all(headers==[1,event,self.batch])
        ids=raw[:,24:self.ids_end].copy().reshape(self.world*self.batch,self.topk)
        probabilities=raw[:,self.ids_end:].copy().view(np.float32).reshape(-1,self.experts)
        layer=event%self.layers;self.history.update(layer,probabilities);gates=self.history.gates(layer)
        status=self.history.native.mgo_route_histogram(ids.ctypes.data,self.hist.ctypes.data,self.world,self.batch,self.topk,self.experts)
        assert status==0
        routes=SimpleNamespace(selected_experts=ids,routing_weights=self.weights,origin_ranks=self.origins,full_router_probs=None)
        return SimpleNamespace(routes=routes,counts=[self.batch]*self.world,histogram=self.hist,gate_scores=gates)
