"""CPU scheduling tests: ready subsets, no wait-to-fill, stable output order."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from mgo_v2.native_expert import NativeExpertExecutor


class NativeSchedulingTests(unittest.TestCase):
    def executor(self, snapshots, cap=128):
        calls, waits, records = [], [], []
        def execute(cache, received, weights, slots, rows, cols):
            calls.append(slots)
            return [('output', slot) for slot in slots]
        rt = SimpleNamespace(cache=None, keys=[0,1,2],
            ready_metrics=dict(waits=0,ready_before_first_wait=0),
            h2d=SimpleNamespace(ready_many=lambda slots: next(snapshots),
                wait_for_slot=waits.append,record_slots_use=records.append))
        ex = NativeExpertExecutor.__new__(NativeExpertExecutor)
        ex.native=SimpleNamespace(execute_wave=execute)
        ex.max_experts=cap;ex.waves=ex.groups=ex.waits=0
        groups=[(i,[],[],i) for i in range(3)]
        return ex,rt,groups,calls,waits,records

    def test_ready_subset_executes_without_waiting_and_preserves_order(self):
        ex,rt,groups,calls,waits,records=self.executor(iter([[False,True,True],[True]]))
        out=ex.compute(rt,('current',None,None,None),dict(groups=groups),0)
        self.assertEqual(calls,[[1,2],[0]])
        self.assertEqual(out,[('output',0),('output',1),('output',2)])
        self.assertEqual(waits,[]);self.assertEqual(records,calls)
        self.assertEqual(rt.ready_metrics,dict(waits=0,ready_before_first_wait=3))

    def test_none_ready_waits_one_slot_without_requiring_new_snapshot(self):
        ex,rt,groups,calls,waits,records=self.executor(iter([[False]*3,[True,True]]))
        ex.compute(rt,('current',None,None,None),dict(groups=groups),0)
        self.assertEqual(calls,[[0],[1,2]]);self.assertEqual(waits,[0])
        self.assertEqual(records,calls)
        self.assertEqual(rt.ready_metrics,dict(waits=1,ready_before_first_wait=0))

    def test_cap_is_upper_bound_and_empty_event_is_safe(self):
        ex,rt,groups,calls,waits,_=self.executor(iter([[True]*3,[True]*2,[True]]),1)
        ex.compute(rt,('current',None,None,None),dict(groups=groups),0)
        self.assertEqual(calls,[[0],[1],[2]]);self.assertEqual(waits,[])
        self.assertEqual(ex.compute(rt,('current',None,None,None),dict(groups=[]),0),[])

    def test_factory_attaches_explicit_native_and_preserves_h0_default(self):
        from mgo_v2.selected_runtime import _create_runtime
        executor=object()
        fake=SimpleNamespace(DecodeOffloadRuntime=lambda *args:SimpleNamespace())
        with patch.dict('sys.modules',{'mgo_v2.decode_runtime':fake}), \
             patch('mgo_v2.native_expert.NativeExpertExecutor',return_value=executor) as factory:
            rt=_create_runtime(SimpleNamespace(),None,None,None,
                dict(expert_executor='native',expert_source='staged'))
            self.assertIs(rt.native_executor,executor)
            factory.assert_called_once()
            rt=_create_runtime(SimpleNamespace(),None,None,None,dict(expert_source='staged'))
            self.assertIsNone(rt.native_executor)
            factory.assert_called_once()

if __name__=='__main__':unittest.main()
