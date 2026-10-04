from adaptive_timing import single_decision,paired_decision

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
 print('PASS adaptive limits, common-mode paired jitter, and unresolved-noise rejection')
if __name__=='__main__':main()
