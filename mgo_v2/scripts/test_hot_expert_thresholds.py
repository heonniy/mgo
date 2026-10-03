import unittest
from hot_expert_thresholds import crossing,N
class ThresholdTests(unittest.TestCase):
    def test_measured_equality_and_no_extrapolation(self):
        costs={n:n/10 for n in N}
        self.assertEqual(crossing(costs,1.6),16)
        self.assertEqual(crossing(costs,1.7),32)
        self.assertIsNone(crossing(costs,100))
    def test_first_measured_nonmonotonic_crossing(self):
        costs={n:1.0 for n in N};costs[8]=3.0;costs[128]=4.0
        self.assertEqual(crossing(costs,2.0),8)
if __name__=='__main__':unittest.main()
