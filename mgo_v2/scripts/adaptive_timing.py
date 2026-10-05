"""Owner-authorized bounded jitter resolution; retain every valid observation."""
import math,statistics
KEYS=('E2E_wall','TPOT')
T95={1:12.7062047364,2:4.30265272975,3:3.18244630528,4:2.7764451052,5:2.57058183564,6:2.44691184879}
def difference(a,b):return abs(a-b)/((a+b)/2)
def stats(values):
 assert len(values)>=2 and all(math.isfinite(x) and x>0 for x in values)
 n=len(values);mean=statistics.mean(values);half=T95[n-1]*statistics.stdev(values)/math.sqrt(n)
 return dict(n=n,mean=mean,median=statistics.median(values),min=min(values),max=max(values),range_relative=difference(min(values),max(values)),mean_CI95=[mean-half,mean+half],relative_CI_half_width=half/mean,early_late_drift=difference(statistics.mean(values[:2]),statistics.mean(values[-2:])))
def single_decision(rows):
 n=len(rows);assert 2<=n<=7
 summary={k:stats([r[k] for r in rows]) for k in KEYS};spread=max(s['range_relative'] for s in summary.values())
 if n==2:target=2 if spread<=.02 else 3 if spread<=.05 else 5
 elif n==3:target=3 if spread<=.05 else 5
 elif n==4:target=5
 else:
  resolved=all(s['relative_CI_half_width']<=.02 and s['early_late_drift']<=.05 for s in summary.values())
  target=5 if resolved and n==5 else 7
 complete=n>=target
 unstable=complete and n>=5 and not all(s['relative_CI_half_width']<=.02 and s['early_late_drift']<=.05 for s in summary.values())
 status='EXTEND_JITTER' if not complete and target>=5 else 'THIRD_REQUIRED' if not complete else 'UNRESOLVED_JITTER' if unstable else 'STABLE_JITTER_ESTIMATE' if n>=5 else 'STABLE'
 estimate={k:(s['mean'] if n!=3 else s['median']) for k,s in summary.items()}
 return dict(status=status,target_repeats=target,complete=complete,unstable=unstable,summary=summary,estimate=estimate,rule='2/3 baseline; noisy cells extend to 5, then at most 7. At n>=5 require mean CI95 half-width <=2% and early/late mean drift <=5%. No sample deletion.')
def legacy_paired_decision(pairs,baseline="BR",candidate="LA"):
 n=len(pairs);assert 2<=n<=7
 single={p:single_decision([r[p] for r in pairs]) for p in (baseline,candidate)};gain={}
 for k in KEYS:
  logs=[math.log(r[baseline][k]/r[candidate][k]) for r in pairs];mean=statistics.mean(logs);half=T95[n-1]*statistics.stdev(logs)/math.sqrt(n)
  ci=[1-math.exp(-(mean-half)),1-math.exp(-(mean+half))];early=1-math.exp(-statistics.mean(logs[:2]));late=1-math.exp(-statistics.mean(logs[-2:]))
  gain[k]=dict(estimate=1-math.exp(-mean),median=statistics.median(1-math.exp(-x) for x in logs),CI95=ci,CI_half_width=(ci[1]-ci[0])/2,early_late_drift=abs(early-late),positive_supported=ci[0]>0,paired_gains=[1-math.exp(-x) for x in logs])
 if n<=3:
  target=max(d['target_repeats'] for d in single.values())
  if any(max(g['paired_gains'])-min(g['paired_gains'])>.02 for g in gain.values()):target=max(target,5)
 elif n==4:target=5
 else:
  resolved=all(g['CI_half_width']<=.02 and g['early_late_drift']<=.02 for g in gain.values());target=5 if resolved and n==5 else 7
 complete=n>=target;unstable=complete and n>=5 and not all(g['CI_half_width']<=.02 and g['early_late_drift']<=.02 for g in gain.values())
 return dict(baseline=baseline,candidate=candidate,status='UNRESOLVED_JITTER' if unstable else 'STABLE_PAIRED_GAIN' if complete else 'EXTEND',complete=complete,unstable=unstable,target_repeats=target,gain=gain,absolute=single,rule='Alternate BR/LA order. Noisy comparisons use 5 or at most 7 complete pairs. Require paired gain CI95 half-width <=2 percentage points and early/late paired-gain drift <=2 points. All observations retained; positive claim requires CI lower bound >0.')

def optimistic_relative_ci(values,final_n):
 """Smallest possible relative t-CI halfwidth with arbitrary positive future times.

 For r remaining samples, the optimum sets every future value to Q/S, where
 S=sum(observed), Q=sum(observed**2). This even grants unrealistically perfect
 future repeats; exceeding the threshold here proves futility within the cap.
 """
 assert 2<=len(values)<=final_n<=7 and all(math.isfinite(x) and x>0 for x in values)
 m=len(values);s=sum(values);q=sum(x*x for x in values);remaining=final_n-m
 return T95[final_n-1]*math.sqrt(max(0.,m*q-s*s)/((final_n-1)*(s*s+remaining*q)))

def tuning_decision(rows):
 """BR-only tuning: retain >=3 samples, prune only mathematically futile extensions."""
 d=single_decision(rows);n=len(rows)
 if n<3 or d['complete'] or d['target_repeats']<5:return d
 finals=[k for k in (5,7) if k>=n]
 bounds={str(final):{key:optimistic_relative_ci([r[key] for r in rows],final) for key in KEYS} for final in finals}
 # At every permitted endpoint, at least one required metric cannot meet 2%.
 if all(any(value>.02+1e-12 for value in bymetric.values()) for bymetric in bounds.values()):
  d.update(status='UNRESOLVED_JITTER_FUTILITY',complete=True,unstable=True,target_repeats=n,optimistic_future_relative_CI_halfwidth=bounds,rule='BR tuning only: retain at least three samples. Stop as ineligible when even optimally chosen future values cannot satisfy the unchanged 2% CI halfwidth limit at either five or seven samples. No samples discarded. Paired BR/LA rules unchanged.')
 return d


def paired_decision(pairs,baseline="BR",candidate="LA"):
 """Owner repeat cap: stable first two stop; at most one third pair."""
 n=len(pairs)
 assert 2<=n<=3, 'new paired measurements are capped at three pairs'
 d=legacy_paired_decision(pairs,baseline,candidate)
 spread=max(s['range_relative'] for policy in d['absolute'].values() for s in policy['summary'].values())
 # At two samples the owner explicitly forbids repeating a stable condition.
 complete=n==3 or spread<=.02
 gain_spread=max(max(g['paired_gains'])-min(g['paired_gains']) for g in d['gain'].values())
 unstable=complete and (spread>(.02 if n==2 else .05) or gain_spread>.02)
 for policy in d['absolute'].values():
  local_spread=max(s['range_relative'] for s in policy['summary'].values())
  policy.update(target_repeats=2 if local_spread<=.02 and n==2 else 3,complete=complete,unstable=complete and local_spread>(.02 if n==2 else .05),rule='Owner cap: stop at two when both metric differences <=2%; otherwise at most three; three-sample full spread must be <=5%.')
 d.update(status='UNRESOLVED_JITTER' if unstable else 'STABLE_PAIRED_GAIN' if complete else 'THIRD_REQUIRED',complete=complete,unstable=unstable,target_repeats=2 if n==2 and complete else 3,rule='Owner repeat amendment: both policies E2E/TPOT differences <=2% stop at two, regardless of CI width; otherwise exactly one third pair, no five/seven or rerun. At three require absolute spread <=5% and paired-gain range <=2 percentage points. Retain all samples and CI; positive claim requires CI lower bound >0.')
 return d
