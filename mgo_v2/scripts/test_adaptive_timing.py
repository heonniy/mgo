from adaptive_timing import single_decision,legacy_paired_decision as paired_decision,tuning_decision,optimistic_relative_ci

def rows(v):return [dict(E2E_wall=x*10,TPOT=x) for x in v]
def main():
 assert single_decision(rows([1,1.01]))['target_repeats']==2
 assert single_decision(rows([1,1.03]))['target_repeats']==3
 assert single_decision(rows([1,1.08]))['target_repeats']==5
 assert single_decision(rows([1,1.08,1.02,1.01,1]))['target_repeats']==7
 d=single_decision(rows([1,1.2,.8,1.1,.9,1.2,.8]));assert d['complete'] and d['unstable']
 pairs=[dict(BR=rows([x])[0],LA=rows([x*.94])[0]) for x in [1,1.2,.8,1.1,.9]]
 d=paired_decision(pairs);assert d['complete'] and not d['unstable'];assert d['gain']['TPOT']['positive_supported'];assert abs(d['gain']['TPOT']['estimate']-.06)<1e-12
 # Alternating noise cannot be hidden by a favorable median or sample deletion.
 pairs=[dict(BR=rows([1])[0],LA=rows([x])[0]) for x in [.8,1.2,.9,1.1,.8,1.2,.9]]
 d=paired_decision(pairs);assert d['complete'] and d['unstable'] and len(d['gain']['TPOT']['paired_gains'])==7
 # Futility bounds match an explicit optimal completion and do not relax gates.
 values=[1.,1.2,.95];best=sum(x*x for x in values)/sum(values)
 bound=optimistic_relative_ci(values,7)
 from adaptive_timing import stats
 assert abs(bound-stats(values+[best]*4)['relative_CI_half_width'])<1e-12
 assert tuning_decision(rows(values))['status']=='UNRESOLVED_JITTER_FUTILITY'
 assert tuning_decision(rows([1.,1.01]))['status']=='STABLE'
 assert tuning_decision(rows([1.,1.03,1.02]))['status']=='STABLE'
 # A noisy prefix with enough possible reduction must still continue.
 assert not tuning_decision(rows([1.,1.055,1.02]))['complete']
 # The paired gain gate is unchanged and can resolve common-mode absolute noise.
 paired=[dict(BR=rows([x])[0],LA=rows([x*.94])[0]) for x in values]
 assert not paired_decision(paired)['complete']
 print('PASS adaptive limits, common-mode paired jitter, and unresolved-noise rejection')
if __name__=='__main__':main()
