#!/usr/bin/env python3
"""Static/CPU-light guards for the R8 phase-aware replay packet."""
import ast
from pathlib import Path
P=Path(__file__).resolve().parents[1]
SRC=(P/'scripts/r8_phase_aware_policy_cpu.py').read_text()
TREE=ast.parse(SRC)

def main():
    # Required policies and stress families.
    for token in ("POLICY_NAMES={0:\"BR\",1:\"CA\",2:\"LA\"}",
                  "BR+REP","CA+REP","LA+REP","COMM","LOAD"):
        assert token in SRC,token
    # Balanced mandatory miss placement and real-slot replicas.
    assert "quota_check.max()-quota_check.min()<=1" in SRC
    assert "choose_slot(best_q" in SRC and "place(best_q" in SRC
    # Every replica creation is separately D2D-accounted; current-miss dependency retained.
    assert "replica_created[event]=1" in SRC
    assert "replica_source_miss[event]=1 if newly_missed[best_e] else 0" in SRC
    assert "dep=hms if r[\"replica_source_miss\"][ev] else 0.0" in SRC
    assert "overlap += max(base,dep+d2d)" in SRC
    assert "serial += base+d2d" in SRC
    # Prefill/decode scopes and all-phase check.
    for token in ('"prefill"','"decode"','"moe_total"',"replica_all_phase=True"):
        assert token in SRC,token
    # B256 membership is the full pool: WORLD*256 == 2048.
    assert "n=WORLD*batch" in SRC
    assert "sample_seeds=[0]" in SRC
    # Phase model must expose both fetch-barrier and overlap/streaming lower-bound views.
    assert "phase_fast_expert_ms" in SRC
    assert "transfer_models" in SRC
    ast.parse((P/'scripts/run_r8_phase_aware_policy_cpu.py').read_text())
    print("PASS phase-aware policy replay guards")

if __name__=='__main__':main()
