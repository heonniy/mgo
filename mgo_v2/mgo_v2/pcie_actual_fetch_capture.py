"""Diagnostic-only log of the actual native controller's mandatory fetches."""
import numpy as np


class ActualFetchCapture:
    def __init__(self,rt):
        assert rt.args.pcie_phase_diagnostic and rt.rank==0
        self.rt=rt;self.original=rt.controller.plan_current;self.rows=[];self.events=0
        def call(event,selected,weights,origins,gate):
            assert event==self.events
            out,promotions,discards=self.original(event,selected,weights,origins,gate)
            assert not promotions and not discards
            fetches=np.array(out[5],dtype=np.int64,copy=True).reshape(-1,5)
            assert len(fetches)==int(out[6][19]) and np.all(fetches[:,4]==0)
            physical=np.array([rt.arena.main_physical[int(rank)][int(slot)] for rank,key,slot,victim,rep in fetches],np.int64)
            self.rows.append(np.column_stack((np.full(len(fetches),event,np.int64),fetches,physical)))
            self.events+=1
            return out,promotions,discards
        rt.controller.plan_current=call

    def finish(self,path):
        self.rt.controller.plan_current=self.original
        rows=np.concatenate(self.rows) if self.rows else np.empty((0,7),np.int64)
        np.savez_compressed(path,fetches=rows,events=np.array(self.events,np.int64),
                            columns=np.array(['event','rank','key','logical_main_slot','victim','replica','physical_slot']))
