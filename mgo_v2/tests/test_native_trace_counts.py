import unittest
import numpy as np
from native_trace_counts import distinct_event_counts


class DistinctTraceCountsTest(unittest.TestCase):
    def test_prefill_and_decode_have_different_event_lengths(self):
        ranks = [np.array(x, dtype=np.uint8) for x in (
            [1, 2, 2, 3, 4, 4],
            [2, 3, 3, 5, 4, 6],
            [1, 7, 7, 3, 8, 8],
            [9, 2, 2, 3, 4, 4],
        )]
        self.assertEqual(distinct_event_counts(ranks, [4, 2]).tolist(), [6, 3])
        with self.assertRaises(ValueError):
            distinct_event_counts(ranks[:3] + [ranks[3][:-1]], [4, 2])


if __name__ == '__main__':
    unittest.main()
