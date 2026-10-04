import numpy as np
from mgo_v2.predictor import transition_counts,normalize,TransitionPredictor,choose_candidates
r=np.array([[[[0,1],[1,2]],[[3,4],[4,5]],[[6,7],[7,8]]]],np.uint8)
c=transition_counts(r);assert c[0,1,4]==2 and c.sum()==16
assert np.array_equal(c,transition_counts(r[:,:,::-1].copy()))
t=normalize(c);assert np.allclose(t.sum(-1)[c.sum(-1)>0],1)
p=TransitionPredictor(t,2);h=np.zeros((2,128));h[0,0]=h[0,1]=1;h[1,1]=h[1,2]=1
y=p.predict_next(0,h);assert y.shape==(2,128) and p.predict_next(2,h) is None
chosen=choose_candidates(y,[128+3],[128+4],1,1,2);assert 3 not in chosen and 4 not in chosen
assert len(choose_candidates(y,[],[],1,0))==0
assert np.array_equal(chosen,choose_candidates(y,[128+3],[128+4],1,1,2))
print('PASS transition normalization, deterministic counts, no-future boundary, resident filtering')

from calibrate_refactor_predictor import histogram
h=histogram(np.array([[127],[0]],np.uint8),np.array([7,0],np.int8))
assert h[7,127]==1 and h[0,0]==1 and h.sum()==2
print("PASS int8 rank histogram overflow regression")
