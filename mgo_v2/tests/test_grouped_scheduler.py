"""B4 batched helpers retain ticket validity and shared overwrite protection."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from mgo_v2.pinned_h2d import PriorityH2DScheduler

class State:
    def __init__(self, state):self.state=state
    def complete(self):
        assert self.state=='INFLIGHT'
        self.state='READY'

class Event:
    def __init__(self, complete=True):self.complete=complete;self.recorded=None
    def query(self):return self.complete
    def record(self, stream):self.recorded=stream

class SchedulerTests(unittest.TestCase):
    def scheduler(self):
        x=PriorityH2DScheduler.__new__(PriorityH2DScheduler)
        x.cv=threading.Condition();x.error=None;x.tickets={};x.compute_done={};x.device='cuda:0'
        return x
    def test_readiness_matches_scalar_with_invalid_and_unsubmitted_slots(self):
        x=self.scheduler()
        for slot,(state,submitted,valid,complete) in enumerate([
            ('READY',True,True,True),('INFLIGHT',True,True,True),
            ('INFLIGHT',True,True,False),('QUEUED',False,True,False),
            ('READY',True,False,True)]):
            x.tickets[slot]=SimpleNamespace(state=State(state),submitted=submitted,valid=valid,done=Event(complete))
        self.assertEqual(x.ready_many([0,1,2,3,4,5]),[True,True,False,False,False,False])
        self.assertEqual(x.ready_many(range(6)),[x.ready(i) for i in range(6)])
        self.assertEqual(x.tickets[1].state.state,'READY')
    def test_wave_event_protects_all_consumed_slots_only(self):
        x=self.scheduler();old=Event();x.compute_done[9]=old;event=Event()
        with patch('torch.cuda.Event',return_value=event) as constructor,patch('torch.cuda.current_stream',return_value='stream'):
            x.record_slots_use([1,4,7])
        constructor.assert_called_once()
        self.assertEqual(event.recorded,'stream')
        for slot in [1,4,7]:self.assertIs(x.compute_done[slot],event)
        self.assertIs(x.compute_done[9],old)

if __name__=='__main__':unittest.main()
