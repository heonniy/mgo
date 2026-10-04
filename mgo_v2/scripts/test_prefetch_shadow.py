import numpy as np
from mgo_v2.prefetch import PrefetchShadow
from mgo_v2.predictor import TransitionPredictor
from env_offload_policy import Policy
T=np.full((47,128,128),1/128);pred=TransitionPredictor(T);caps=np.full(8,32,np.int32)
sel=np.tile(np.arange(8,dtype=np.uint8),(8,1));weights=np.ones((8,8),np.float32)/8;orig=np.arange(8,dtype=np.int8);gate=np.zeros(128,np.float32)
for policy,kind in [('BR',0),('CA',1),('LA',4)]:
 p=PrefetchShadow(caps,0,policy,42,pred);ref=Policy(caps,np.zeros((48,128,128),np.float32),False,kind,42)
 # Separate BR calls share the compiled RNG; reseed each independent one-event test.
 out,pro,dis=p.plan_current(0,sel,weights,orig,gate)
 from env_offload_policy import seed_rng
 seed_rng(42);expected=ref.apply(0,sel,weights,orig,gate,np.zeros((128,8),np.int32))
 assert out[5]==expected[5] and np.array_equal(p.main.slots,ref.slots) and not pro and not dis
 p=PrefetchShadow(caps,1,policy,42,pred);p.plan_current(0,sel,weights,orig,gate);assert p.plan_prefetch_next(np.ones((8,128)))==[]
 p.plan_current(48,sel,weights,orig,gate);planned=p.plan_prefetch_next(np.ones((8,128)))
 assert len(planned)==8 and len({key for _,key,_ in planned})==8
 out,promotions,discard=p.plan_current(49,sel,weights,orig,gate)
 assert len(promotions)==8 and not discard and not out[5]
 assert not p.pending and p.counters['useful']==8
print('PASS P0 admission parity, prefill-off, balanced unique prefetch and promotion reuse')
