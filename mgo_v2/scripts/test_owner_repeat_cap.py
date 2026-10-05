"""Owner limit applies to stable cells even with inconclusive gain intervals."""
from adaptive_timing import paired_decision

def pair(br,la):return {p:dict(E2E_wall=x*10,TPOT=x) for p,x in [('BR',br),('LA',la)]}
def main():
 d=paired_decision([pair(1,.94),pair(1.01,.95)])
 assert d['complete'] and d['target_repeats']==2 and not d['unstable']
 # Individually stable observations cannot trigger a CI-driven third sample.
 d=paired_decision([pair(1,1),pair(1.01,.99)])
 assert d['complete'] and d['target_repeats']==2
 d=paired_decision([pair(1,.94),pair(1.04,.98)])
 assert not d['complete'] and d['target_repeats']==3
 d=paired_decision([pair(1,.94),pair(1.04,.98),pair(1.02,.96)])
 assert d['complete'] and not d['unstable'] and d['target_repeats']==3
 d=paired_decision([pair(1,.94),pair(1.2,.8),pair(1.1,.9)])
 assert d['complete'] and d['unstable'] and d['target_repeats']==3
 try:paired_decision([pair(1,.94)]*4)
 except AssertionError:pass
 else:raise AssertionError('fourth pair incorrectly accepted')
 print('PASS owner two/three cap and unresolved-noise exclusion')
if __name__=='__main__':main()
