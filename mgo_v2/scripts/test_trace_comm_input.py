import unittest
from trace_comm_input import classify,validate_events

class TraceCommTests(unittest.TestCase):
    def test_gate_boundaries(self):
        for ratios,want in [([1.2,1.2],'STRONG_GAP'),([1.1,1.19],'AMBIGUOUS_GAP'),
                            ([.95,1.5],'NO_GAP'),([.9,.98],'NO_GAP'),([1,1.1],'NO_GAP'),
                            ([1,1.5],'AMBIGUOUS_GAP'),([1.11,1.29],'STRONG_GAP')]:
            self.assertEqual(classify(ratios),want)
    def test_batch_scaled_counts(self):
        for batch in (4,16,32):
            events=[dict(source_event=i+48,layer=i%48,
                         dispatch_send=[[batch]*4 for _ in range(4)],dispatch_recv=[[batch]*4 for _ in range(4)],
                         combine_send=[[2*batch]*4 for _ in range(4)],combine_recv=[[2*batch]*4 for _ in range(4)]) for i in range(384)]
            validate_events(events,batch)
            with self.assertRaises(AssertionError):validate_events(events,8)

    def test_corrupt_receive_rejected(self):
        e=[dict(source_event=i+48,layer=i%48,dispatch_send=[[8]*4 for _ in range(4)],
                dispatch_recv=[[8]*4 for _ in range(4)],combine_send=[[16]*4 for _ in range(4)],
                combine_recv=[[16]*4 for _ in range(4)]) for i in range(384)]
        validate_events(e)
        e[17]['combine_recv'][1][2]+=1
        with self.assertRaises(AssertionError):validate_events(e)

if __name__=='__main__':unittest.main()
