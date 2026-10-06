"""Rebuild only H1b's hash-verified B3 reference signatures; no H2 lookahead."""
import json,hashlib
from pathlib import Path
import torch
from mgo_v2.graph_expert import GraphExpertExecutor


def prepare(rt,model,ids,mask,teacher,generate,validate,place_threads,write):
 root=Path('/home/hwlee/mgo-results/critical_path_admission_followup_20261006/b3/C30')
 prior=root/f'B3_C30_NODE1_H1b_V3_OPT_PF_OVERLAP_{rt.args.policy}_B128_H8'
 rank=rt.rank;old=json.loads((prior/f'b3_correctness_rank{rank}.json').read_text());assert old['status']=='PASS'
 packet=Path(__file__).resolve().parents[1]/'experiments/critical_path_admission_followup_20261006'
 receipt=json.loads((packet/'B3_H1b_CORRECTNESS.json').read_text())
 expected=next(r for r in receipt['rows'] if r['policy']==rt.args.policy and r['rank']==rank)
 assert old['reference']==expected['reference']
 sigrows=json.loads((packet/'B3_H1b_GRAPH_SIGNATURES.json').read_text())['rows']
 signature=next(r for r in sigrows if r['policy']==rt.args.policy and r['rank']==rank)
 rawsig=json.loads((prior/f'b3_signatures_rank{rank}.json').read_text())
 assert signature['signatures']==rawsig['signatures']
 rt.stage_frozen_inputs(64);graph=GraphExpertExecutor(rt.cache,rt.kernel,wrapper=True);graph.signatures=set(map(tuple,rawsig['signatures']));rt.graph_executor=graph
 rt.h2d.close();graph.build(lambda row:write(rt.args.output/f'b3_preparation_rank{rank}.json',dict(stage='H1b_REFERENCE_CAPTURE',**row)))
 write(rt.args.output/f'b3_signatures_rank{rank}.json',graph.receipt())
 rt.reset();place_threads();rt.graph_check=True
 row,tokens=generate(model,rt,ids,mask,teacher,64)
 full=Path('/home/hwlee/mgo-results/policy_regime_20261005/C30/inputs_B128_H64')
 proof=json.loads((full/f'{rt.args.policy}_P2_proof.json').read_text());validate(rt,row,proof,rank)
 actual=dict(controller=dict(rt.controller.counters),copies=rt.h2d.metrics['copies'],bytes=rt.h2d.metrics['bytes'],canceled=rt.h2d.metrics['canceled'],forward_bytes=rt.transport.forward_bytes,return_bytes=rt.transport.return_bytes,calls=rt.transport.calls,token_hash=row['argmax_hash'])
 assert actual==old['reference'],'H1b full64 reference changed'
 write(rt.args.output/f'b3_correctness_rank{rank}.json',dict(status='PASS',horizon=64,reference=old['reference'],observed=actual,graph=graph.receipt(),reused_signatures_only_for_H1b=True))
 rt.graph_check=False;rt.reset();place_threads()
