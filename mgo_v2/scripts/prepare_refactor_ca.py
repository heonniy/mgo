"""Prepare secondary LA/CA pairs without retuning the timing-selected arm."""
import hashlib
import json
from pathlib import Path

from prepare_refactor_arms import PACKET, groups_for


def prepare(timing, frozen):
    if timing['status'] != 'TIMING_CANDIDATE':
        raise ValueError('Complete stable three-arm timing evidence is required')
    winner = timing['winner']
    rows = [row for row in timing['rows'] if row['arm'] == winner]
    if len(rows) != 1 or not rows[0]['eligible'] or not rows[0]['stable']:
        raise ValueError('Secondary CA cannot run on an ineligible winner')
    groups = [g for g in groups_for(frozen, timing.get("horizon",256), [128, 256]) if g['runtime_arm'] == winner]
    assert len(groups) == 2
    for group in groups:
        if timing.get('world')==4:group.update(world=4,gpus=[0,1,4,5])
        for case, policy in zip(group['cases'], ('LA', 'CA')):
            case.update(label=policy, policy=policy)
    return groups


def main(a):
    timing_path = a.timing
    frozen_path = PACKET / 'M13_FROZEN_PREFETCH.json'
    timing_raw, frozen_raw = timing_path.read_bytes(), frozen_path.read_bytes()
    timing, frozen = json.loads(timing_raw), json.loads(frozen_raw)
    groups = prepare(timing, frozen)
    target = PACKET / (a.prefix+'M17_CA_CONFIG.json')
    receipt = PACKET / (a.prefix+'M17_CA_PREPARATION.json')
    if target.exists() or receipt.exists():
        raise FileExistsError('Refusing to replace a frozen CA configuration')
    target.write_text(json.dumps(groups, indent=2) + '\n')
    receipt.write_text(json.dumps(dict(
        status='PREPARED', physically_validated=False, chosen_arm=timing['winner'],
        primary_selection_uses_CA=False,
        timing_selection_sha256=hashlib.sha256(timing_raw).hexdigest(),
        prefetch_selection_sha256=hashlib.sha256(frozen_raw).hexdigest(),
        config_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        interpretation='Secondary paired LA/CA characterization on the frozen timing winner. LA is baseline and CA is candidate, so positive secondary gain means CA is faster. All three primary arms and both batches must be measured first. Profiling and final selection receipts remain separately required.',
    ), indent=2) + '\n')


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--timing',type=Path,default=PACKET/'THREE_ARM_RESULTS.json');p.add_argument('--prefix',default='');main(p.parse_args())
