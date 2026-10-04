#!/usr/bin/env python3
"""Static/CPU-light guards for the calibrated R8 phase-aware replay packet."""
import ast
from pathlib import Path

P=Path(__file__).resolve().parents[1]
SRC=(P/'scripts/r8_phase_aware_policy_cpu.py').read_text()
MICRO=(P/'examples/replica_phase_microbench.py').read_text()
TREE=ast.parse(SRC)

def main():
    # Required policies and separate stress families.
    for token in ('POLICY_NAMES={0:"BR",1:"CA",2:"LA"}',
                  '"BR+REP"','"CA+REP"','"LA+REP"','"COMM"','"LOAD"'):
        assert token in SRC,token

    # Mandatory misses are balanced before any replica decision.
    assert 'assert h2d_counts[event].max()-h2d_counts[event].min()<=1' in SRC
    assert 'allow_replica=replica_enabled' in SRC

    # Real-slot persistent replicas and current-miss dependency.
    assert 'slot=choose_slot(q,layer,active,slots,capacities,last,gates,True,EXPERTS)' in SRC
    assert 'place(best_q,key,best_slot,True' in SRC
    assert 'replica_created[event]=1' in SRC
    assert 'replica_source_miss[event]=1 if newly_missed[best_e] else 0' in SRC

    # Env1/Env2 measured calibration is mandatory.
    assert 'microbench_calibration.json' in SRC
    assert 'set(micro["environments"])=={"env1","env2"}' in SRC
    assert 'phase_overlap_archived_expert_ms' in SRC
    assert 'phase_serial_archived_expert_ms' in SRC
    assert 'robust phase gain=min(Env1 overlap, Env1 serial, Env2 overlap, Env2 serial)' in SRC

    # Prefill/decode scopes and B256 full-pool semantics.
    for token in ('"prefill"','"decode"','"moe_total"',"replica_all_phase=True"):
        assert token in SRC,token
    assert 'n=WORLD*batch' in SRC and 'sample_seeds=[0]' in SRC

    # Microbench must contain activation G2G, D2D, overlap and strict serial.
    for token in ("activation_oneway","activation_roundtrip","d2d_expert_onepair",
                  "d2d_expert_twopair","resident_d2d_overlap_h2d",
                  "miss_h2d_then_d2d_overlap_next_h2d",
                  "miss_h2d_h2d_d2d_serial","expert_compute"):
        assert token in MICRO,token

    ast.parse((P/'scripts/run_r8_phase_aware_policy_cpu.py').read_text())
    ast.parse((P/'scripts/run_replica_phase_microbench.py').read_text())
    ast.parse((P/'scripts/summarize_replica_phase_microbench.py').read_text())
    ast.parse((P/'scripts/run_replica_calibrated_cpu_packet.py').read_text())
    print("PASS calibrated phase-aware policy replay guards")

if __name__=='__main__':
    main()
