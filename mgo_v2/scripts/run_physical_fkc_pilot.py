#!/usr/bin/env python3
"""Sequential six-cell launcher; fixed order, no retries, shared-host guards."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_rank_oracle_study import memory, group_rss, process_tree

PACKAGE = Path(__file__).resolve().parents[1]
PACKET = PACKAGE / 'experiments/fetch_comm_pareto_p2p_20261002'
ROOT = Path('/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002/physical_fkc_20261003')
SMOKE = Path('/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002/exact_payload_20261003/crossover_worker.py')
ORDER = [('T0', 'F'), ('R3', 'C'), ('T0', 'K'), ('R3', 'K'), ('T0', 'C'), ('R3', 'F')]


def write(path, obj):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2) + '\n'); tmp.replace(path)


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def validate_cell(target, mode, point, prior):
    metadata = json.loads((PACKET / 'physical_fkc_schedule_metadata.json').read_text())
    expected = metadata['schedules'][point]
    rs = [json.loads((target / f'rank{r}.json').read_text()) for r in range(4)]
    for rank, result in enumerate(rs):
        assert result['status'] == 'PASS' and result['rank'] == rank
        assert result['point'] == point and result['mode'] == mode
        assert result['schedule_sha256'] == expected['sha256'] and len(result['events']) == 432
        assert result['generated_tokens_equal_source'] and result['fetch_modes'][1] == 0
        if prior: assert result['generated_tokens'] == prior[rank]
    for i in range(432):
        records = [r['events'][i] for r in rs]
        assert len({r['post_state_sha256'] for r in records}) == 1
        for a in range(4):
            assert records[a]['event'] == i and records[a]['physical_cache_parity'] and records[a]['raw_route_parity']
            for b in range(4):
                for phase in ('dispatch', 'combine'):
                    assert records[a][phase + '_send_counts'][b] == records[b][phase + '_recv_counts'][a]
    for phase, span in (('full', range(432)), ('prefill', range(48)), ('decode', range(48, 432))):
        counts = dict(first_copy_fetches=0, reload_fetches=0, replica_fetches=0, peer_activation_bytes=0, remote_token_rank_pairs=0)
        for rank, result in enumerate(rs):
            for i in span:
                event = result['events'][i]
                for k in ('first_copy_fetches', 'reload_fetches', 'replica_fetches'): counts[k] += event['h2d_classes'][k]
                remote_dispatch = sum(n for dst, n in enumerate(event['dispatch_send_counts']) if dst != rank)
                remote_combine = sum(n for dst, n in enumerate(event['combine_send_counts']) if dst != rank)
                counts['peer_activation_bytes'] += (remote_dispatch + remote_combine) * 4096
                counts['remote_token_rank_pairs'] += remote_dispatch
                before = result['events'][i - 1]['physical_fetches'] if i else 0
                assert event['physical_fetches'] - before == sum(event['h2d_classes'].values())
        counts['total_fetches'] = sum(counts[k] for k in ('first_copy_fetches', 'reload_fetches', 'replica_fetches'))
        counts['expert_h2d_bytes'] = counts['total_fetches'] * 9 * 1024**2
        assert counts == expected[phase], (phase, counts, expected[phase])
    assert sum(r['physical_fetches'] for r in rs) == expected['full']['total_fetches']
    activation = sum(sum(r['collectives']['peer_payload_tx_bytes'].get(k, 0) for k in ('dispatch_hidden', 'return_outputs')) for r in rs)
    assert activation == expected['full']['peer_activation_bytes']
    return [r['generated_tokens'] for r in rs]


def run(label, mode, point=None):
    target = ROOT / label; target.mkdir(exist_ok=False)
    smoke = point is None
    env = dict(os.environ, PYTHONPATH=str(PACKAGE) + ':' + str(PACKAGE / 'examples'),
               CUDA_VISIBLE_DEVICES='0,1,4,5', OMP_NUM_THREADS='1' if smoke else '4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MGO_TRANSPORT=mode, MOE_EP_DISABLE_ARCHER_EVICT='1', MOE_EP_NATIVE_NUMERICS='1', MOE_EP_SLOT_VIEWS='1')
    for k in list(env):
        if k.startswith('NCCL_'): del env[k]
    if mode == 'R3': env.update(NCCL_P2P_LEVEL='LOC', NCCL_IB_DISABLE='1')
    if smoke: env.update(NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,SHM', NCCL_DEBUG_FILE=str(target / 'nccl-%h-%p.log'))
    initial = memory()
    assert initial['host_available_bytes'] >= 512 * 2**30 and all(g['used_mib'] < 1024 for g in initial['gpu'].values()), initial
    script = SMOKE if smoke else PACKAGE / 'examples/physical_fkc_worker.py'
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4', str(script), '--output', str(target)]
    cmd += ['--smoke'] if smoke else ['--mode', mode, '--point', point]
    state = dict(status='RUNNING', mode=mode, point=point, smoke=smoke, command=cmd, memory=[initial],
                 transport_env={k:v for k,v in env.items() if k.startswith('NCCL_')}, started_unix=time.time(),
                 worker_sha256=sha(script))
    write(target / 'status.json', state)
    reason = None
    with (target / 'run.log').open('w') as log:
        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state['pid'] = process.pid
        while process.poll() is None:
            time.sleep(5)
            sample = memory(); sample['group_rss_bytes'] = group_rss(process.pid)
            state['memory'].append(sample); write(target / 'status.json', state)
            if time.time() - state['started_unix'] > (120 if smoke else 900): reason = 'bounded timeout'
            if sample['host_available_bytes'] < 128 * 2**30 or sample['group_rss_bytes'] > (32 if smoke else 320) * 2**30 or any(g['free_mib'] < 8192 for g in sample['gpu'].values()): reason = 'memory guard'
            if reason:
                descendants, _ = process_tree(process.pid)
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    for pid in descendants:
                        try: os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                    process.wait()
                break
    state.update(status='PASS' if process.returncode == 0 and reason is None else 'FAIL', exit_code=process.returncode,
                 stop_reason=reason, finished_unix=time.time())
    write(target / 'status.json', state)
    assert state['status'] == 'PASS', str(target / 'run.log')
    if smoke:
        for r in range(4):
            receipt = json.loads((target / f'rank{r}.json').read_text())
            assert receipt['status'] == 'PASS' and receipt['results'][0]['payload_valid']
        lines = [line for p in target.glob('nccl-*.log') for line in p.read_text().splitlines() if 'via ' in line]
        kind = 'SHM/direct/direct' if mode == 'R3' else 'P2P/'
        assert lines and any(kind in line for line in lines)
        assert not any('via NET/' in line for line in lines)
        if mode == 'R3': assert not any('via P2P/' in line for line in lines)
        (PACKET / f'physical_fkc_{mode}_transport.log').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(label=label, status='PASS', seconds=state['finished_unix']-state['started_unix'])), flush=True)
    return target


def main():
    lock = (ROOT / 'pilot.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = PACKET / 'physical_fkc_progress.json'
    sources = [Path(__file__), PACKAGE / 'examples/physical_fkc_worker.py', PACKAGE / 'scripts/frozen_replica_schedule.py']
    sources += sorted((PACKAGE / 'mgo_v2').glob('*.py'))
    sources += list((PACKAGE.parent / 'MoE-Infinity-EP-archer-coslot/moe_infinity').glob('_store*.so'))
    state = dict(status='RUNNING', order=['-'.join(x) for x in ORDER], completed_cells=[], preflights=[],
                 source_sha256={str(p): sha(p) for p in sources}, source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PACKAGE,text=True).strip())
    write(manifest_path, state)
    try:
        for mode in ('T0', 'R3'):
            run('preflight_' + mode, mode); state['preflights'].append(mode); write(manifest_path, state)
        tokens = None
        for mode, point in ORDER:
            state['active_cell'] = mode + '-' + point; write(manifest_path, state)
            target = run(mode + '_' + point, mode, point)
            tokens = validate_cell(target, mode, point, tokens)
            state['completed_cells'].append(mode + '-' + point); write(manifest_path, state)
        state.update(status='PASS', active_cell=None)
    except BaseException as exc:
        state.update(status='FAIL', error=repr(exc)); write(manifest_path, state); raise
    write(manifest_path, state)


if __name__ == '__main__': main()
