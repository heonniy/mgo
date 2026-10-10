"""Audit the matched-token continuation R8 quota control."""

import hashlib
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009')
POLICIES = (('near_frozen','LA_CA_NEAR','Original Near'),
            ('fast_frozen','NEAR_FAST','Fast-rank quota'),
            ('pcie_frozen','NEAR_PCIE','PCIe lookup quota'))
METRICS = ('TTFT','TPOT','E2E','throughput')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(label, policy, name, teacher, teacher_sha, table_sha):
    first = RAW / 'jobs' / f'{label}_full_v1'
    status = json.loads((first / 'status.json').read_text())
    assert status['status']=='PASS' and status['repeats']==2
    assert status['source_commit']== '57cee06f07b8b71820940959902e13bfdafa6834'
    assert status['physical_gpus']==list(range(8)) and status['quiet_2367']
    assert status['workload_sha256']==teacher['workload_sha256']
    command = status['command']
    assert command[command.index('--policy')+1]==policy
    assert '--prefetch-off' in command and '--expert-executor' in command
    path = Path(command[command.index('--teacher-tokens')+1])
    assert sha(path)==teacher_sha
    if policy!='LA_CA_NEAR':
        assert sha(Path(command[command.index('--quota-table')+1]))==table_sha
    runs = [json.loads((first/f'repeat{i}.json').read_text()) for i in (1,2)]
    gap={key:100*abs(runs[0][key]-runs[1][key])/statistics.mean(x[key] for x in runs)
         for key in ('TPOT','E2E')}
    paths=[first]
    if max(gap.values())>2:
        extra=RAW/'jobs'/f'{label}_full_v2'
        extra_status=json.loads((extra/'status.json').read_text())
        assert extra_status['status']=='PASS' and extra_status['repeats']==1
        assert extra_status['source_commit']==status['source_commit']
        assert extra_status['workload_sha256']==status['workload_sha256']
        paths.append(extra)
        runs.append(json.loads((extra/'repeat1.json').read_text()))
    else:
        assert not (RAW/'jobs'/f'{label}_full_v2').exists()
    audits=[]
    for source in paths:
        repeats=json.loads((source/'status.json').read_text())['repeats']
        for i in range(1,repeats+1):
            ranks=[json.loads((source/f'repeat{i}_rank{rank}.json').read_text())
                   for rank in range(8)]
            assert all(x['forced_continuation'] and x['no_compile']
                       and x['prefetch_off'] and x['expert_executor']=='native'
                       and x['policy']==policy
                       and x['validation']['status']=='PASS'
                       and x['validation']['physical_slots']==1843
                       and x['validation']['controller']['quota_violations']==0
                       for x in ranks)
            diffs=[sum(a!=b for actual, expected in zip(x['tokens'],teacher['rank_tokens'][str(rank)])
                       for a,b in zip(actual,expected))
                   for rank,x in enumerate(ranks)]
            audits.append(dict(path=str(source),repeat=i,
                               actual_vs_teacher_token_differences=diffs,
                               mandatory_misses=ranks[0]['validation']['controller']['mandatory'],
                               h2d_copies=[x['validation']['scheduler']['copies'] for x in ranks],
                               h2d_bytes=[x['validation']['scheduler']['bytes'] for x in ranks],
                               argmax_hashes=[x['argmax_hash'] for x in ranks]))
    metrics={}
    for metric in METRICS:
        samples=[float(x[metric]) for x in runs]
        metrics[metric]=dict(samples=samples,
                             reported=statistics.median(samples) if len(samples)==3
                             else statistics.mean(samples),
                             minimum=min(samples),maximum=max(samples))
    return dict(name=name,policy=policy,paths=list(map(str,paths)),
                first_pair_gap_pct=gap,metrics=metrics,audits=audits)


def fmt(x,key):
    m=x['metrics'][key];p=4 if key=='TPOT' else 3
    return f"{m['reported']:.{p}f} [{m['minimum']:.{p}f}, {m['maximum']:.{p}f}]"


def main():
    assert json.loads((HERE/'FROZEN_STATUS.json').read_text())['status']=='PASS'
    teacher_path=HERE/'FROZEN_TOKENS.json'
    teacher=json.loads(teacher_path.read_text())
    table_path=HERE/'LOOKUP.json'
    rows=[row(label,policy,name,teacher,sha(teacher_path),sha(table_path))
          for label,policy,name in POLICIES]
    result=dict(status='PASS',scope='matched generated-token continuation, not frozen internal routing',
                teacher_sha256=sha(teacher_path),lookup_sha256=sha(table_path),policies=rows)
    (HERE/'FROZEN_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Matched-token continuation control','',
           'Same R8/C30/B16/input512/output64 Qwen ShareGPT requests, cache '
           'budget, Ready-First native executor, Near placement and prefetch '
           'OFF. All policies consume the same precomputed next-token IDs '
           'from the original Near target. Actual argmax outputs are still '
           'computed and recorded. This controls generated-token input drift '
           'but BF16 internal hidden states and router choices can differ.', '',
           '| Policy | Repeats | TTFT (s) | TPOT (s/token) | E2E (s) | TPS | Actual token differences vs teacher |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        token_diffs=sum(x['audits'][0]['actual_vs_teacher_token_differences'])
        lines.append(f"| {x['name']} | {len(x['metrics']['E2E']['samples'])} | "
                     f"{fmt(x,'TTFT')} | {fmt(x,'TPOT')} | {fmt(x,'E2E')} | "
                     f"{fmt(x,'throughput')} | {token_diffs} |")
    lines+=['','The table reports the mean of two or median of three unfiltered '
            'repeats, with the full range. The token-difference count is from '
            'the first target and does not change the common next-token inputs. '
            'Per-repeat token differences, rank H2D copy counts, cache state '
            'and raw paths are in [FROZEN_RESULTS.json](FROZEN_RESULTS.json).','']
    (HERE/'FROZEN_RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
