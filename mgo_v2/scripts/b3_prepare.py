"""Untimed complete-schedule discovery, capture, and H0/H1 validation."""
import json
import numpy as np
import torch
from mgo_v2.graph_expert import GraphExpertExecutor


def prepare(rt, model, ids, mask, teacher, generate, validate, place_threads, write):
    from pathlib import Path
    from torch._dynamo.utils import counters
    rank = rt.rank
    output = rt.args.output
    full = Path('/home/hwlee/mgo-results/policy_regime_20261005') / rt.args.b3_cache / 'inputs_B128_H64'
    proof = json.loads((full / f'{rt.args.policy}_P2_proof.json').read_text())
    def status(stage, **kw):
        write(output / f'b3_preparation_rank{rank}.json', dict(stage=stage, **kw))
    rt.stage_frozen_inputs(64)
    executor = GraphExpertExecutor(rt.cache, rt.kernel, wrapper=getattr(rt.args,'b3_wrapper',False))
    rt.graph_executor = executor
    status('FULL64_H0_DISCOVERY')
    reference, expected = generate(model, rt, ids, mask, teacher, 64)
    validate(rt, reference, proof, rank)
    baseline = dict(controller=dict(rt.controller.counters), copies=rt.h2d.metrics['copies'],
                    bytes=rt.h2d.metrics['bytes'], canceled=rt.h2d.metrics['canceled'],
                    forward_bytes=rt.transport.forward_bytes, return_bytes=rt.transport.return_bytes,
                    calls=rt.transport.calls, token_hash=reference['argmax_hash'])
    rt.h2d.synchronize(); torch.cuda.synchronize();rt.h2d.close()
    status('GRAPH_CAPTURE', signatures=len(executor.signatures))
    executor.build(lambda row: status('GRAPH_CAPTURE', **row))
    write(output / f'b3_signatures_rank{rank}.json', executor.receipt())
    rt.reset(); place_threads(); rt.graph_check = True
    status('FULL64_H1_VALIDATION')
    before = dict(counters['stats'])
    with torch._dynamo.config.patch(error_on_recompile=True):
        row, actual = generate(model, rt, ids, mask, teacher, 64)
    validate(rt, row, proof, rank)
    assert before == dict(counters['stats']), 'compilation during H1 validation'
    assert np.array_equal(actual, expected), 'H0/H1 token mismatch'
    observed = dict(controller=dict(rt.controller.counters), copies=rt.h2d.metrics['copies'],
                    bytes=rt.h2d.metrics['bytes'], canceled=rt.h2d.metrics['canceled'],
                    forward_bytes=rt.transport.forward_bytes, return_bytes=rt.transport.return_bytes,
                    calls=rt.transport.calls, token_hash=row['argmax_hash'])
    assert observed == baseline, ('H0/H1 counter mismatch', baseline, observed)
    assert executor.checked > 0 and len(executor.entries) == len(executor.signatures)
    write(output / f'b3_correctness_rank{rank}.json', dict(status='PASS', horizon=64,
          reference=baseline, observed=observed, graph=executor.receipt(),
          exact_expert_output_prefix_steps=1, no_compile_in_validation=True))
    rt.graph_check = False
    rt.reset(); place_threads()
    status('PREPARED', signatures=len(executor.signatures))
