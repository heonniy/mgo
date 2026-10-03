"""Small compiled proxy check, run only after diagnostic timing finishes."""
import numpy as np
from ca_stress_cpu import proxy_scores
from ca_stress_common import PACKET, write

routes = np.empty((1, 4, 8), np.uint8)
routes[0, :2] = np.arange(8)
routes[0, 2:] = np.arange(8, 16)
orders = np.array([[0, 1, 2, 3], [0, 2, 1, 3]], np.int64)
result = proxy_scores(routes, np.arange(4, dtype=np.int64), orders, 2, 2)
assert np.array_equal(result, [16., 0.]), result
assert proxy_scores.nopython_signatures
write(PACKET / 'compiled_proxy_check.json', dict(
    status='PASS', clustered_score=float(result[0]), mixed_score=float(result[1]),
    nopython_compiled=True, GPU_used=False))
