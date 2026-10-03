"""Predeclared per-pass checks for actual-minus-self remote premiums."""
RHOS=(0,.125,.25,.5,.75)

def timing_gate(rows):
    checks=[]
    for pass_id in (0,1):
        data={r['rho']:r for r in rows if r['pass_index']==pass_id}
        assert set(data)==set(RHOS)
        premiums={rho:r['actual_ms']-r['self_ms'] for rho,r in data.items()}
        for rho in RHOS[:-1]:checks.append(dict(pass_index=pass_id,check='nonnegative_remote',rho=rho,value=premiums[rho],passed=premiums[rho]>=0))
        base=premiums[0]
        checks.append(dict(pass_index=pass_id,check='defined_rho0_normalization',rho=0,value=base,passed=base>0))
        for prev,nxt in zip(RHOS[:-2],RHOS[1:-1]):
            normalized_prev=premiums[prev]/base if base>0 else None
            normalized_next=premiums[nxt]/base if base>0 else None
            passed=base>0 and premiums[prev]>=0 and premiums[nxt]>=0 and premiums[nxt]<=1.1*premiums[prev]
            checks.append(dict(pass_index=pass_id,check='adjacent_normalized_premium',rho=nxt,previous_rho=prev,normalized_previous=normalized_prev,normalized_next=normalized_next,passed=passed))
        a,s=data[.75]['actual_ms'],data[.75]['self_ms']
        checks.append(dict(pass_index=pass_id,check='zero_remote_control_agreement',rho=.75,actual_ms=a,self_ms=s,relative_difference=(a-s)/s if s>0 else None,passed=s>0 and abs(a-s)<=.1*s))
    return dict(passed=all(r['passed'] for r in checks),checks=checks)
