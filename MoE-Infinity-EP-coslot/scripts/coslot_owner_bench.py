"""Owner-policy benchmark — Qwen3-235B EP=4, naive vs balanced.

Controlled comparison (run twice with --owner naive / balanced, SAME --seed →
identical sample_ids + initial condition):
  * ShareGPT first-human prompts, fixed-seed random 32 samples, each ≥48 tokens
    (truncated to exactly 48; shorter ones excluded + resampled).
  * rank r gets samples[r*8:(r+1)*8]  (batch=8/rank, distinct per rank).
  * manual prefill (1 forward) + exactly 64 decode steps (no HF generate → no
    eos handling; exactly 64 new tokens).
  * measures: prefill/TTFT, decode total, TPOT, prefill/decode a2a,
    fetch/compute/combine (archer phase timers), per-rank routed tokens +
    imbalance, drift / NaN.
Writes per-run results JSON to MOE_EP_BENCH_OUT.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

from moe_infinity_ep.launch.distributed_setup import pin_visible_device

pin_visible_device()

from moe_infinity_ep.launch.distributed_setup import init_distributed, barrier  # noqa: E402
from moe_infinity_ep.launch.entry import MoE_EP  # noqa: E402
from moe_infinity_ep.launch import init_logging as ilog  # noqa: E402
from moe_infinity_ep.utils.config import Config  # noqa: E402

INPUT_LEN = int(os.environ.get("MOE_EP_INPUT_LEN", "48"))
OUTPUT_LEN = int(os.environ.get("MOE_EP_OUTPUT_LEN", "64"))
BATCH_PER_RANK = int(os.environ.get("MOE_EP_BENCH_BATCH_PER_RANK", "8"))
DATASET = os.environ.get("MOE_EP_DATASET", "sharegpt")  # "sharegpt" | "lmsys"


def _sample_prompts(ds, tokenizer, seed, n_total):
    """Deterministic: shuffle by seed, take first-human prompts that tokenize
    to >= INPUT_LEN tokens, truncate to INPUT_LEN.  Returns (ids, texts, tok)."""
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    sample_ids, texts, tok_rows = [], [], []
    for i in idxs:
        row = ds[i]
        if DATASET == "lmsys":
            convs = row["conversation"]   # [{role, content}]
            human = next((c["content"] for c in convs
                          if c.get("role") == "user" and c.get("content")), None)
            rid = row["conversation_id"]
        else:
            convs = row["conversations"]  # [{from, value}]
            human = next((c["value"] for c in convs
                          if c.get("from") == "human" and c.get("value")), None)
            rid = row["id"]
        if not human:
            continue
        ids = tokenizer(human, add_special_tokens=True).input_ids
        if len(ids) < INPUT_LEN:
            continue                      # too short → exclude, resample
        sample_ids.append(rid)
        texts.append(human)
        tok_rows.append(ids[:INPUT_LEN])  # truncate to exactly INPUT_LEN
        if len(sample_ids) == n_total:
            break
    if len(sample_ids) < n_total:
        raise RuntimeError(f"only {len(sample_ids)}/{n_total} prompts >= "
                           f"{INPUT_LEN} tok")
    return sample_ids, texts, tok_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--owner", required=True,
                    choices=["naive", "balanced", "random", "ours"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    world = int(os.environ["WORLD_SIZE"])
    cfg.resolve_parallel(world)
    topology = init_distributed(cfg.parallel.ep_size)
    rank = topology.global_rank

    import torch
    import torch.distributed as dist
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path, trust_remote_code=True)
    if DATASET == "lmsys":
        import glob as _glob
        arrow = sorted(_glob.glob(
            "/home/work/hyewon.lee/dataset/lmsys___lmsys-chat-1m/**/*.arrow",
            recursive=True))
    else:
        arrow = ("/home/work/hyewon.lee/dataset/Aeala___share_gpt_vicuna_unfiltered/"
                 "default/0.0.0/8b0048ad6ae8c22f46a78c15559dec98feef5539/"
                 "share_gpt_vicuna_unfiltered-train.arrow")
    ds = load_dataset("arrow", data_files=arrow, split="train")
    bsz = BATCH_PER_RANK * world                 # concurrent requests per batch
    n_eval = int(os.environ.get("MOE_EP_N_EVAL", str(bsz)))
    n_batches = max(1, n_eval // bsz)
    n_eval = n_batches * bsz                      # round down to full batches
    n_total = n_eval                              # total eval samples (streamed)
    sample_ids, texts, tok_rows = _sample_prompts(ds, tokenizer, args.seed, n_eval)
    if topology.is_rank0:
        print(f"[bench] owner={args.owner} seed={args.seed} world={world} "
              f"bs/rank={BATCH_PER_RANK} in={INPUT_LEN} out={OUTPUT_LEN} "
              f"cap={cfg.offload.cache_capacity_per_rank}", flush=True)
        print(f"[bench] sample_ids(all {n_total})={sample_ids}", flush=True)

    t0 = time.time()
    engine = MoE_EP(cfg, topology)
    engine.load()
    if topology.is_rank0:
        print(f"[bench] load {time.time()-t0:.1f}s", flush=True)

    model = engine.model
    model.eval()
    exec_ = engine.engine.expert_executor
    ld = exec_.local_dispatcher
    counters = exec_.counters

    def sync_barrier():
        torch.cuda.synchronize()
        barrier()

    nan_seen = False
    ctrl = getattr(exec_, "controller", None)
    tok_dir = os.environ.get("MOE_EP_DECODE_TOKENS_DIR")
    # Reset CUMULATIVE stats ONCE: placement_stats + counters accumulate over ALL
    # batches, so hit/remote/sum_max_fetch reflect the full streamed eval (cache
    # persists across batches = realistic warm-up).  walls/phase are summed below.
    if ctrl is not None and hasattr(ctrl, "reset_placement_stats"):
        ctrl.reset_placement_stats()
    exec_._routed_prefill = 0
    exec_._routed_decode = 0
    exec_._fetch_ops_rank_prefill = []
    exec_._fetch_ops_rank_decode = []
    exec_._miss_modulo_decode = []
    exec_._fetch_layer_imbal_decode = []

    prefill_wall = 0.0
    decode_wall = 0.0
    prefill_phase = [0, 0, 0, 0]
    decode_phase = [0, 0, 0, 0]
    prefill_mode = [0, 0]
    decode_mode = [0, 0]

    def _accphase(dst, src):
        for i in range(min(len(dst), len(src))):
            dst[i] += int(src[i])

    # ---------- streamed eval: n_batches × (prefill + OUTPUT_LEN decode) ----------
    for b in range(n_batches):
        lo = b * bsz
        my = slice(lo + rank * BATCH_PER_RANK, lo + (rank + 1) * BATCH_PER_RANK)
        input_ids = torch.tensor(tok_rows[my], dtype=torch.long).cuda()
        attn = torch.ones_like(input_ids)
        if ctrl is not None and hasattr(ctrl, "reset_rank_accum"):
            ctrl.reset_rank_accum()                  # new batch of requests
        # ---- prefill ----
        ld.reset_phase_times()
        sync_barrier()
        tp0 = time.time()
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
        sync_barrier()
        prefill_wall += time.time() - tp0
        logits = out.logits[:, -1, :]
        nan_seen |= bool(torch.isnan(logits).any().item())
        past = out.past_key_values
        next_tok = logits.argmax(-1, keepdim=True)
        _accphase(prefill_phase, ld.get_phase_times().tolist())
        _accphase(prefill_mode, ld.get_fetch_mode_counts().tolist())
        # ---- decode (teacher-forced per batch for fair cross-policy demand) ----
        tok_file = (f"{tok_dir}/decode_in_b{BATCH_PER_RANK}_bt{b}_rank{dist.get_rank()}.pt"
                    if tok_dir else None)
        fixed_in = None
        if tok_file and os.path.exists(tok_file):
            fixed_in = torch.load(tok_file, map_location="cpu").cuda()  # [B, OUTPUT_LEN]
        inputs_fed = []
        ld.reset_phase_times()
        cur_attn = attn
        sync_barrier()
        td0 = time.time()
        with torch.no_grad():
            for step in range(OUTPUT_LEN):
                if fixed_in is not None:
                    next_tok = fixed_in[:, step:step + 1]   # identical demand
                inputs_fed.append(next_tok)
                cur_attn = torch.cat(
                    [cur_attn, torch.ones(cur_attn.size(0), 1, dtype=cur_attn.dtype,
                                          device=cur_attn.device)], dim=1)
                out = model(input_ids=next_tok, attention_mask=cur_attn,
                            past_key_values=past, use_cache=True)
                past = out.past_key_values
                lg = out.logits[:, -1, :]
                if torch.isnan(lg).any().item():
                    nan_seen = True
                next_tok = lg.argmax(-1, keepdim=True)
        sync_barrier()
        decode_wall += time.time() - td0
        if tok_file and fixed_in is None:           # reference run: persist tokens
            torch.save(torch.cat(inputs_fed, dim=1).cpu(), tok_file)
        _accphase(decode_phase, ld.get_phase_times().tolist())
        _accphase(decode_mode, ld.get_fetch_mode_counts().tolist())
        if (ctrl is not None and hasattr(ctrl, "commit_batch_to_collection")
                and os.environ.get("MOE_EP_OURS_NO_ONLINE", "0") != "1"):
            ctrl.commit_batch_to_collection()       # online history growth (gated)
        if topology.is_rank0:
            ch = ""
            ps = getattr(ctrl, "placement_stats", None)
            if ps:
                db = ps["decode"]; tot = db["n_hits"] + db["n_miss"]
                if tot:
                    ch = f" cum_hit={db['n_hits']/tot:.3f}"
            print(f"[bench] batch {b+1}/{n_batches} done "
                  f"(cum decode_wall={decode_wall:.2f}s{ch})", flush=True)
    # routing-forcing: explicitly persist recorded routing (atexit is unreliable
    # under torchrun worker teardown).
    _rf = getattr(exec_, "_routing_forcer", None)
    if _rf is not None:
        _rf.save()
    routed_prefill = int(exec_._routed_prefill)
    routed_decode = int(exec_._routed_decode)

    # ---------- cross-rank gather (routed tokens + nan + per-rank fetch) ----------
    # per-rank PCIe fetch workload: fetch_wait us + direct/staging counts are
    # genuinely per-rank (each rank's own archer), so all_gather them.
    # decode_phase = [fetch, weightcopy, compute, combine] (us, this rank).
    # Gather the FULL per-phase array across ranks so the straggler (the rank
    # that actually bounds wall-clock) is visible — not just rank0 (which can be
    # underloaded under an imbalanced policy, making rank0-only timers misleading).
    cdr = counters.to_dict()   # this rank's counter sums (a2a is per-rank wall)
    rp = torch.tensor([routed_prefill, routed_decode, 1 if nan_seen else 0,
                       int(prefill_phase[0]), int(decode_phase[0]),
                       int(prefill_mode[0]), int(prefill_mode[1]),
                       int(decode_mode[0]), int(decode_mode[1]),
                       int(decode_phase[1]), int(decode_phase[2]),
                       int(decode_phase[3] if len(decode_phase) > 3 else 0),
                       int(cdr.get("prefill_a2a_forward_us_sum", 0)),
                       int(cdr.get("prefill_a2a_backward_us_sum", 0)),
                       int(cdr.get("decode_a2a_forward_us_sum", 0)),
                       int(cdr.get("decode_a2a_backward_us_sum", 0))],
                      dtype=torch.long).cuda()
    gathered = [torch.zeros_like(rp) for _ in range(world)]
    dist.all_gather(gathered, rp)
    routed_pf = [int(g[0].item()) for g in gathered]
    routed_dec = [int(g[1].item()) for g in gathered]
    any_nan = any(int(g[2].item()) for g in gathered)
    fetch_wait_pf = [int(g[3].item()) for g in gathered]
    fetch_wait_dec = [int(g[4].item()) for g in gathered]
    direct_pf = [int(g[5].item()) for g in gathered]
    staging_pf = [int(g[6].item()) for g in gathered]
    direct_dec = [int(g[7].item()) for g in gathered]
    staging_dec = [int(g[8].item()) for g in gathered]
    wcopy_dec = [int(g[9].item()) for g in gathered]      # decode weight-copy per rank
    compute_dec = [int(g[10].item()) for g in gathered]   # decode GEMM per rank
    combine_dec = [int(g[11].item()) for g in gathered]   # decode combine per rank
    a2a_pf_fwd = [int(g[12].item()) for g in gathered]    # per-rank a2a wall (us)
    a2a_pf_bwd = [int(g[13].item()) for g in gathered]
    a2a_dec_fwd = [int(g[14].item()) for g in gathered]
    a2a_dec_bwd = [int(g[15].item()) for g in gathered]

    sync_barrier()
    if topology.is_rank0:
        cd = counters.to_dict()
        drift = max(counters.drift_layer_pre) if counters.drift_layer_pre else 0

        def imbalance(xs):
            m = sum(xs) / len(xs)
            std = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
            return {"per_rank": xs, "max": max(xs), "min": min(xs),
                    "mean": round(m, 1),
                    "max_over_mean": round(max(xs) / m, 4) if m else 0.0,
                    "std_over_mean": round(std / m, 4) if m else 0.0}

        # ---- per-rank PCIe FETCH WORKLOAD (balanced's actual objective) ----
        # fetch_ops counts come from the GLOBAL controller plan (identical on
        # every rank → rank0 sees all ranks).  Each fetch is one expert of a
        # fixed byte size, so fetch_bytes is exactly count × expert_bytes.
        fetch_ops_pf = exec_._fetch_ops_rank_prefill or [0] * world
        fetch_ops_dec = exec_._fetch_ops_rank_decode or [0] * world
        mc = model.config
        inter = getattr(mc, "moe_intermediate_size", None) or mc.intermediate_size
        # gate + up + down = 3 matrices of [hidden, moe_intermediate], dtype bytes
        dt_bytes = torch.finfo(model.dtype).bits // 8
        expert_bytes = 3 * mc.hidden_size * inter * dt_bytes
        li = exec_._fetch_layer_imbal_decode
        li_sorted = sorted(li)
        fetch_workload = {
            "expert_bytes": expert_bytes,
            "prefill": {
                "fetch_ops": imbalance(fetch_ops_pf),
                "fetch_bytes_per_rank": [c * expert_bytes for c in fetch_ops_pf],
                "fetch_wait_us_per_rank": fetch_wait_pf,
                "direct_per_rank": direct_pf, "staging_per_rank": staging_pf,
                "total_direct": sum(direct_pf), "total_staging": sum(staging_pf),
            },
            "decode": {
                "fetch_ops": imbalance(fetch_ops_dec),
                "fetch_bytes_per_rank": [c * expert_bytes for c in fetch_ops_dec],
                "fetch_wait_us_per_rank": fetch_wait_dec,
                "direct_per_rank": direct_dec, "staging_per_rank": staging_dec,
                "total_direct": sum(direct_dec), "total_staging": sum(staging_dec),
                # naive places miss expert e on rank e%ep — the modulo dist
                # shows whether e%ep is already (near-)uniform for naive.
                "miss_expert_modulo": exec_._miss_modulo_decode or [0] * world,
                # per-layer fetch_ops imbalance (max/mean over ranks), decode.
                "layer_imbal_mean": round(sum(li) / len(li), 4) if li else 0.0,
                "layer_imbal_max": round(max(li), 4) if li else 0.0,
                "layer_imbal_p95": round(
                    li_sorted[int(0.95 * (len(li) - 1))], 4) if li else 0.0,
                "n_layers_sampled": len(li),
            },
        }

        total_in = INPUT_LEN * n_total
        res = {
            "owner": args.owner, "seed": args.seed, "world": world,
            "batch_per_rank": BATCH_PER_RANK, "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN, "cap_per_rank": cfg.offload.cache_capacity_per_rank,
            "sample_ids": sample_ids,
            # latency / throughput (s, tok/s)
            "prefill_latency_s": round(prefill_wall, 4),
            "ttft_s": round(prefill_wall, 4),
            "prefill_throughput_tok_s": round(total_in / prefill_wall, 1),
            "decode_total_latency_s": round(decode_wall, 4),
            "e2e_latency_s": round(prefill_wall + decode_wall, 4),  # prefill+decode wall (all batches)
            "n_eval": n_eval, "n_batches": n_batches,
            "tpot_ms": round(decode_wall / (OUTPUT_LEN * n_batches) * 1000, 3),
            # a2a (rank0, us summed over layers)
            "prefill_a2a_us": cd["prefill_a2a_forward_us_sum"] + cd["prefill_a2a_backward_us_sum"],
            "decode_a2a_us": cd["decode_a2a_forward_us_sum"] + cd["decode_a2a_backward_us_sum"],
            "prefill_a2a_fwd_us": cd["prefill_a2a_forward_us_sum"],
            "prefill_a2a_bwd_us": cd["prefill_a2a_backward_us_sum"],
            "decode_a2a_fwd_us": cd["decode_a2a_forward_us_sum"],
            "decode_a2a_bwd_us": cd["decode_a2a_backward_us_sum"],
            "total_a2a_us": (cd["prefill_a2a_forward_us_sum"] + cd["prefill_a2a_backward_us_sum"]
                             + cd["decode_a2a_forward_us_sum"] + cd["decode_a2a_backward_us_sum"]),
            # archer phase (rank0, us): [fetch_wait, weight_copy, gemm, combine]
            # weight_copy = per-expert slot->param_ D2D; compute = pure GEMM.
            "prefill_fetch_us": prefill_phase[0],
            "prefill_weightcopy_us": prefill_phase[1] if len(prefill_phase) > 3 else 0,
            "prefill_compute_us": prefill_phase[2] if len(prefill_phase) > 3 else prefill_phase[1],
            "prefill_combine_us": prefill_phase[-1],
            "decode_fetch_us": decode_phase[0],
            "decode_weightcopy_us": decode_phase[1] if len(decode_phase) > 3 else 0,
            "decode_compute_us": decode_phase[2] if len(decode_phase) > 3 else decode_phase[1],
            "decode_combine_us": decode_phase[-1],
            # local_exec (=fetch+compute+combine wall via Python timer, rank0)
            "prefill_local_exec_us": cd["prefill_local_exec_us_sum"],
            "decode_local_exec_us": cd["decode_local_exec_us_sum"],
            # inter-layer gap (decode): combine-A2A done -> next-layer MoE start
            # (= attention/non-MoE block between consecutive MoE layers, rank0).
            "decode_interlayer_gap_us": float(getattr(exec_, "_interlayer_gap_us_decode", 0.0)),
            "decode_interlayer_gap_n": int(getattr(exec_, "_interlayer_gap_n", 0)),
            # routing imbalance (execution-side; NOT balanced's objective)
            "routed_prefill": imbalance(routed_pf),
            "routed_decode": imbalance(routed_dec),
            # PCIe fetch workload (balanced's actual objective)
            "fetch_workload": fetch_workload,
            # correctness
            "drift_max": int(drift), "nan_seen": bool(any_nan),
            # per-rank decode phase (us) + straggler(max) — wall-clock is bounded
            # by the SLOWEST rank, so report per-rank + max (not just rank0).
            "decode_phase_per_rank": {
                "fetch_us": fetch_wait_dec, "weightcopy_us": wcopy_dec,
                "compute_us": compute_dec, "combine_us": combine_dec},
            "decode_phase_max": {
                "fetch_us": max(fetch_wait_dec), "weightcopy_us": max(wcopy_dec),
                "compute_us": max(compute_dec), "combine_us": max(combine_dec)},
            # per-rank a2a wall (us): collective이라 rank별 도착시각 차이 =
            # straggler 대기가 어느 rank에 쌓이는지 직접 보임.
            "a2a_per_rank": {
                "prefill_fwd_us": a2a_pf_fwd, "prefill_bwd_us": a2a_pf_bwd,
                "decode_fwd_us": a2a_dec_fwd, "decode_bwd_us": a2a_dec_bwd},
            "cache_hits": cd.get("cache_hits_total", counters.cache_hits),
            "cache_misses": counters.cache_misses,
            # placement diagnostics (MOE_EP_PLACEMENT_STATS=1): token movement
            # (local/remote = a2a/NVLink volume) + victim-in-demand (staging cause)
            "placement_stats": getattr(
                getattr(exec_, "controller", None), "placement_stats", None),
            # churn: expert당 재-fetch(thrash) + rank별 고유 expert (locality가 왜
            # 더 churn하는지 진단).
            "churn": (lambda c: c.churn_summary() if c is not None and
                      hasattr(c, "churn_summary") else None)(
                getattr(exec_, "controller", None)),
            # retrieval discriminability: top-1 cosine distribution (all-high ==
            # undiscriminative ~ frequency; spread == matching adds signal).
            "retrieval": (lambda r: None if r is None else {
                "col_size": len(r.col),
                "n": len(r.cos_top1),
                "cos_mean": round(sum(r.cos_top1) / len(r.cos_top1), 4) if r.cos_top1 else None,
                "cos_min": round(min(r.cos_top1), 4) if r.cos_top1 else None,
                "cos_max": round(max(r.cos_top1), 4) if r.cos_top1 else None,
                "cos_p10": round(sorted(r.cos_top1)[len(r.cos_top1)//10], 4) if r.cos_top1 else None,
                "cos_p90": round(sorted(r.cos_top1)[min(len(r.cos_top1)-1, 9*len(r.cos_top1)//10)], 4) if r.cos_top1 else None,
            })(getattr(getattr(exec_, "controller", None), "_retrieval", None)),
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"[bench] RESULT owner={args.owner} ttft={res['ttft_s']}s "
              f"tpot={res['tpot_ms']}ms drift={res['drift_max']} "
              f"nan={res['nan_seen']} routed_dec_imbal="
              f"{res['routed_decode']['max_over_mean']}", flush=True)
        print(f"[bench] wrote {args.out}", flush=True)

    barrier()
    if topology.is_rank0:
        print("[bench] OK", flush=True)
    torch.cuda.synchronize()
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    ilog.proc_mem("pre_exit(normal)")
    os._exit(0)


if __name__ == "__main__":
    _rc = 1
    try:
        _rc = int(main() or 0)
    except BaseException as _e:
        import traceback
        traceback.print_exc()
        print(f"[bench] FATAL: {_e!r}", flush=True)
        _rc = 1
    finally:
        ilog.proc_mem(f"pre_exit(rc={_rc})")
        os._exit(_rc)
