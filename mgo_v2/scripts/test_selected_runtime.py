"""Configuration gate only; fixtures are not physical selection evidence."""
import json
import runpy
import tempfile
from pathlib import Path

module = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'mgo_v2/selected_runtime.py'))
options = module['selected_options']
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'selection.json'
    data = dict(status='PASS', frozen=True, precision='bf16',
                chosen_arm='V2_OPT_PF_BARRIER', prefetch=dict(P=2, trigger='T2'))
    path.write_text(json.dumps(data))
    default = options(path)
    assert default['arena_budget'] == 2 and not default['streaming']
    for arm in module['ARMS']:
        row = options(path, arm)
        assert row['arena_budget'] == (0 if arm == module['ARMS'][0] else 2)
        assert row['trigger'] == 'T2' and row['partial_precision'] == 'bf16'
        assert row['streaming'] == row['ready_first'] == (arm == module['ARMS'][2])
        assert row['physical_prefetch'] and row['fused']
    for key, value in [('status', 'TIMING_CANDIDATE'), ('frozen', False), ('precision', 'fp32'), ('chosen_arm', 'unknown')]:
        broken = {**data, key:value}
        path.write_text(json.dumps(broken))
        try:
            options(path)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid selection accepted: {key}')
    path.unlink()
    try:
        options(path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Missing selection silently enabled a default')
print('PASS selected default/ablations and provisional/missing/non-BF16 rejection')
