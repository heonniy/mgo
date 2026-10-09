"""Compare prespecified prior-winner controls and separately selected new winners."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
from pcie_host import write

PKG=Path(__file__).resolve().parents[1];REPO=PKG.parent


def report(out,commit):
    historical=json.loads((PKG/'experiments/pcie_topology_ablation_20261009/maxgain64/final/cohort/maxgain_validation.json').read_text())
    assert historical['status']=='PASS'
    old=historical['best_observed']
    cells=[dict(local_batch=16,global_requests=64,output_tokens=64,status='HISTORICAL_PASS',
         primary_repeats=5,best=old,control=old,source='maxgain64/final/cohort/maxgain_validation.json',
         timing_context='Historical concurrent administration disclosed; all repetitions retained')]
    for batch in (16,32,64):
        for n in (64,128,256):
            if (batch,n)==(16,64):continue
            path=out/f'B{batch}_O{n}/final/cohort/maxgain_validation.json'
            # Archive compresses large JSON losslessly.
            compressed=path.with_name(path.name+'.gz')
            if path.exists():data=json.loads(path.read_text())
            elif compressed.exists():
                import gzip
                with gzip.open(compressed,'rt') as f:data=json.load(f)
            else:
                cells.append(dict(local_batch=batch,global_requests=4*batch,output_tokens=n,status='PENDING'));continue
            assert data['status']=='PASS' and data['local_batch']==batch and data['output_tokens']==n
            control=next(c for c in data['comparisons'] if c['candidate']=='previous_best_control')
            cells.append(dict(local_batch=batch,global_requests=batch*4,output_tokens=n,status='PASS',
                best=data['best_observed'],control=control,source=str(path.relative_to(out)),
                primary_repeats=data['arms'][data['best_observed']['candidate']+'/R-NEAR']['live']['primary_repeats']))
    rows=sorted(cells,key=lambda r:(r['local_batch'],r['output_tokens']))
    write(out/'MATRIX_RESULTS.json',dict(status='PASS' if all(c['status']!='PENDING' for c in rows) else 'INCOMPLETE',cells=rows,
        scope='Observed screened winners; no global optimum. Prespecified control and searched winners are separate comparisons'))
    with (out/'matrix.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['local_batch','global_requests','output_tokens','decode_forwards','status','repeats','control_R_TPOT','control_G_TPOT','control_gain_percent','best_candidate','best_R_TPOT','best_G_TPOT','best_gain_percent'])
        for c in rows:
            fixed=c.get('control');best=c.get('best')
            w.writerow([c['local_batch'],c['global_requests'],c['output_tokens'],c['output_tokens']-1,c['status'],c.get('primary_repeats'),
                *([fixed['R']['median'],fixed['G']['median'],fixed['gain_percent']] if fixed else [None]*3),
                *([best['candidate'],best['R']['median'],best['G']['median'],best['gain_percent']] if best else [None]*4)])
    lines=['# Best-input batch and output-length expansion','',
        'Local batch is per GPU/rank; global requests are four times local batch. Output includes the first prefill-produced token, so decode forwards = output tokens minus one. EOS never stops generation.',
        '', 'The previous winner is a prespecified control. B16 keeps its exact64 ordered inputs; B32/B64 add distinct whole-corpus requests and repartition across ranks. Newly searched winners can differ by cell and must not be treated as matched-input scaling.',
        '', 'All retained primary repeats determine medians. The short-routing proxy and single-pair screening scores are selection evidence only. Finalists are frozen before repeats. Positive gain is not assumed; separate diagnostics and token agreement expose changed live trajectories.',
        '', '| Local/global batch | Output | State | Control R / G TPOT (s) | Control gain | Searched best | Best gain |',
        '|---|---:|---|---:|---:|---|---:|']
    for c in rows:
        fixed=c.get('control');best=c.get('best')
        lines.append(f"| {c['local_batch']}/{c['global_requests']} | {c['output_tokens']} | {c['status']} | "+
            (f"{fixed['R']['median']:.6f} / {fixed['G']['median']:.6f} | {fixed['gain_percent']:.3f}% | {best['candidate']} | {best['gain_percent']:.3f}% |" if fixed else '— | — | — | — |'))
    lines+=['','Historical B16/O64 retained the disclosed concurrent CPU/Git administration. New primary windows perform no builds, analytical generation or Git packing. GPU lease/burn transitions and NUMA-shared pinned108GiB source proofs are archived per job.','',
            'Each completed cell retains exact source IDs/manifests hashes, cold-cache/full-trace parity, assigned experts, all-rank serialized phase validation, memory, every primary and separate TPOT breakdown. Pending cells have no result claim.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if commit:
        paths=[out/name for name in ('MATRIX_RESULTS.json','matrix.csv','RESULTS.md')]
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
        subprocess.run(['git','add','--',*[str(p.relative_to(REPO)) for p in paths]],cwd=REPO,check=True)
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-q','-m','Update measured best-input batch/output matrix; keep pending cells and fixed controls explicit'],cwd=REPO,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',action='store_true')
    a=p.parse_args();report(a.out,a.commit)
