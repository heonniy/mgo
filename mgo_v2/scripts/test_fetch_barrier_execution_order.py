#!/usr/bin/env python3
"""Static guard for the physical fetch-barrier execution contract.

This intentionally avoids importing the GPU worker so it can run on CPU-only
hosts and in CI.
"""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORKER=(ROOT/'examples/env_offload_worker.py').read_text()
RUNNER=(ROOT/'scripts/run_env_offload_cell.py').read_text()

def between(text,left,right):
    lo=text.index(left);hi=text.index(right,lo)
    return text[lo:hi]

def main():
    execute=between(WORKER," def execute(self,layer,hidden,selected,weights,probs):","  self.index+=1;return output")
    # Physical fetch-barrier order must remain explicit and serial.
    markers=[
        "e=self.plan_event(",
        "with nvtx_phase('moe.dispatch')",
        "torch.cuda.synchronize()",
        "with nvtx_phase('moe.h2d_fetch')",
        "with nvtx_phase('moe.h2d_global_barrier')",
        "with nvtx_phase('moe.expert_compute')",
        "with nvtx_phase('moe.combine')",
    ]
    pos=[execute.index(m) for m in markers]
    assert pos==sorted(pos),pos

    plan=between(WORKER," def plan_event(self,layer,selected,weights,probs):"," def apply_fetches(self,e):")
    assert plan.index("moe.metadata_exchange") < plan.index("moe.rank_decision")
    assert "fetch-barrier physical validation requires --schedule-mode live" in WORKER
    assert "dist.barrier();torch.cuda.synchronize()" in WORKER
    assert "self.current_global_fetch_count=len(fetches)" in WORKER
    assert "if self.current_global_fetch_count:" in execute

    # Historical baseline remains available and runner wires both knobs through.
    assert "choices=['streaming','fetch-barrier']" in WORKER
    assert "choices=['frozen','live']" in WORKER
    assert "'--execution-order',execution_order" in RUNNER
    assert "'--schedule-mode',schedule_mode" in RUNNER
    print("PASS fetch-barrier execution-order static guard")

if __name__=="__main__":main()
