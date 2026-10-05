"""Default optimized runtime selected by completed physical evidence.

No provisional winner or unverified fallback is silently enabled. Explicit
ablation modes reuse the same frozen prefetch choice and BF16 runtime stack.
"""
import json
from pathlib import Path

ARMS = ('V1_OPT_NOPF_BARRIER', 'V2_OPT_PF_BARRIER', 'V3_OPT_PF_OVERLAP')
DEFAULT_SELECTION = Path(__file__).resolve().parents[1] / 'experiments/decode_prefetch_runtime_refactoring_20261004/FINAL_RUNTIME_SELECTION.json'


def arm_options(arm, budget, trigger):
    if arm not in ARMS or trigger not in ('T0', 'T1', 'T2'):
        raise ValueError('Unknown runtime arm or trigger')
    if budget not in ((0,) if arm == ARMS[0] else (1, 2, 4)):
        raise ValueError('Invalid arena budget for the runtime arm')
    overlap = arm == ARMS[2]
    return dict(runtime_arm=arm, arena_budget=budget, trigger=trigger,
                partial_precision='bf16', streaming=overlap, ready_first=overlap,
                physical_prefetch=True, fused=True)


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
    options=arm_options(chosen, 0 if chosen == ARMS[0] else prefetch['P'], prefetch['trigger'])
    if 'staging_backend' in selection:
        if selection['staging_backend'] not in ('torch','memmove'):raise ValueError('Unknown selected staging backend')
        options['staging_backend']=selection['staging_backend']
    if 'unique_combine' in selection:
        if type(selection['unique_combine']) is not bool:raise ValueError('Invalid unique combine flag')
        options['unique_combine']=selection['unique_combine']
    if 'async_metadata_inputs' in selection:
        if type(selection['async_metadata_inputs']) is not bool:raise ValueError('Invalid async metadata flag')
        options['async_metadata_inputs']=selection['async_metadata_inputs']
    return options


def _create_runtime(args, model, backing, experts, options):
    for name, value in options.items():
        setattr(args, name, value)
    from .decode_runtime import DecodeOffloadRuntime
    return DecodeOffloadRuntime(args, model, backing, experts)


def create_explicit_runtime(args, model, backing, experts, case):
    """Construct an explicit experimental arm using the default's same path."""
    options = arm_options(case['runtime_arm'], case['P'], case['trigger'])
    options['staging_backend']=case.get('staging_backend','torch')
    options['unique_combine']=case.get('unique_combine',False)
    options['async_metadata_inputs']=case.get('async_metadata_inputs',False)
    if type(options['async_metadata_inputs']) is not bool:raise ValueError('Invalid async metadata flag')
    if type(options['unique_combine']) is not bool:raise ValueError('Invalid unique combine flag')
    if options['staging_backend'] not in ('torch','memmove'):raise ValueError('Unknown staging backend')
    if case['partial_precision'] != 'bf16' or case['overlap'] != options['streaming']:
        raise ValueError('Case disagrees with the common BF16 runtime arm')
    return _create_runtime(args, model, backing, experts, options)


def create_selected_runtime(args, model, backing, experts, *, selection_path=None, arm=None):
    """Use the measured default, or explicitly select either ablation mode.

    The caller supplies the existing physical harness's model, expert backing,
    frozen-input arguments and placement policy. No second cache is created.
    """
    options = selected_options(selection_path, arm)
    return _create_runtime(args, model, backing, experts, options)
