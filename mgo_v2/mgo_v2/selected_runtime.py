"""Default optimized runtime selected by completed physical evidence.

No provisional winner or unverified fallback is silently enabled. Explicit
ablation modes reuse the same frozen prefetch choice and BF16 runtime stack.
"""
import json
from pathlib import Path

ARMS = ('V1_OPT_NOPF_BARRIER', 'V2_OPT_PF_BARRIER', 'V3_OPT_PF_OVERLAP')
DEFAULT_SELECTION = Path(__file__).resolve().parents[1] / 'experiments/decode_prefetch_runtime_refactoring_20261004/FINAL_RUNTIME_SELECTION.json'


def selected_options(selection_path=None, arm=None):
    path = Path(selection_path) if selection_path is not None else DEFAULT_SELECTION
    selection = json.loads(path.read_text())
    if selection.get('status') != 'PASS' or selection.get('frozen') is not True:
        raise ValueError('A completed, frozen physical runtime selection is required')
    if selection.get('precision') != 'bf16':
        raise ValueError('The selected runtime requires BF16')
    winner = selection['chosen_arm']
    if winner not in ARMS:
        raise ValueError(f'Unknown selected runtime arm: {winner}')
    chosen = winner if arm is None else arm
    if chosen not in ARMS:
        raise ValueError(f'Unknown ablation arm: {chosen}')
    prefetch = selection['prefetch']
    if prefetch['P'] not in (1, 2, 4) or prefetch['trigger'] not in ('T0', 'T1', 'T2'):
        raise ValueError('Invalid frozen BR-only prefetch configuration')
    overlap = chosen == ARMS[2]
    return dict(runtime_arm=chosen, arena_budget=0 if chosen == ARMS[0] else prefetch['P'],
                trigger=prefetch['trigger'], partial_precision='bf16',
                streaming=overlap, ready_first=overlap, physical_prefetch=True, fused=True)


def create_selected_runtime(args, model, backing, experts, *, selection_path=None, arm=None):
    """Use the measured default, or explicitly select either ablation mode.

    The caller supplies the existing physical harness's model, expert backing,
    frozen-input arguments and placement policy. No second cache is created.
    """
    options = selected_options(selection_path, arm)
    for name, value in options.items():
        setattr(args, name, value)
    from .decode_runtime import DecodeOffloadRuntime
    return DecodeOffloadRuntime(args, model, backing, experts)
