"""Real-archer stress + DATA-CORRECTNESS diagnosis under fetch/staging pressure.

Loads once, runs K fixed rounds (collective-safe — every rank runs the same
number of rounds, no early break) on a fixed per-rank prompt.  Per round it
captures the per-step LOGITS (not just argmax tokens) so we can tell apart:

  * benign FP/NCCL nondeterminism — bf16 GEMM + all-to-all + index_add are not
    bit-exact run-to-run; argmax flips only on a near-tie (tiny top-2 margin).
  * real stream-race CORRUPTION — fetch/staging puts wrong bytes in a slot →
    large, structured logit divergence (≫ the bf16 noise floor), or argmax
    flips where the top-2 margin is LARGE.

Round-to-round the cache WARM STATE differs (round k starts with round k-1's
residue), so identical logits across rounds also proves the result is a pure
function of the inputs — independent of cache/fetch/evict ordering.

Checks every round: drift (controller Phase-4 verify raises mid-gen on any
shadow≠physical), fetch==miss, staging exercised, and the logit diff vs round0.

Env: STRESS_ROUNDS (5), STRESS_TOKENS (8).
"""
from __future__ import annotations

import os
import sys
import time

from moe_infinity_ep.launch.distributed_setup import pin_visible_device

pin_visible_device()

from moe_infinity_ep.launch.distributed_setup import init_distributed, barrier  # noqa: E402
from moe_infinity_ep.launch.entry import MoE_EP  # noqa: E402
from moe_infinity_ep.launch import init_logging as ilog  # noqa: E402
from moe_infinity_ep.utils.config import Config  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Photosynthesis converts",
    "The author of 'Pride and Prejudice' is",
    "The chemical symbol for gold is",
    "The speed of light is approximately",
    "The currency of Japan is",
    "The longest river in the world is",
]


def _counts(ld):
    fm = ld.get_fetch_mode_counts().tolist()      # [direct, staging]
    cs = ld.get_cache_stats().tolist()            # [visit,hit,miss,fetch,evict,ovl]
    return fm[0], fm[1], cs[1], cs[2], cs[3], cs[4]


def main() -> int:
    import torch
    cfg_path = sys.argv[1]
    rounds = int(os.environ.get("STRESS_ROUNDS", "5"))
    tokens = int(os.environ.get("STRESS_TOKENS", "8"))

    cfg = Config.from_yaml(cfg_path)
    world_size = int(os.environ["WORLD_SIZE"])
    cfg.resolve_parallel(world_size)
    topology = init_distributed(cfg.parallel.ep_size)
    rank = topology.global_rank
    r0 = topology.is_rank0

    if r0:
        print(f"[stress] world={topology.world_size} ep={topology.ep_size} "
              f"cap={cfg.offload.cache_capacity_per_rank or 'auto'} "
              f"rounds={rounds} tokens={tokens} "
              f"verify={os.environ.get('MOE_EP_VERIFY_LAYER_END','1')}", flush=True)

    t0 = time.time()
    engine = MoE_EP(cfg, topology)
    engine.load()
    if r0:
        print(f"[stress] load {time.time()-t0:.1f}s", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.path, trust_remote_code=True)
    model = engine.model
    model.eval()
    ld = engine.engine.expert_executor.local_dispatcher

    canary = PROMPTS[rank % len(PROMPTS)]
    enc = tok(canary, return_tensors="pt")
    ids = enc.input_ids.cuda()
    attn = enc.attention_mask.cuda()

    ref_scores = None        # [T, vocab] fp32 from round0
    noise_floor = 0.0        # max PRE-divergence |logit diff| = pure FP noise
                             # (only steps where argmax still matches ref; after
                             # the first flip the sequence cascades and diffs are
                             # meaningless for corruption detection).
    diverge_recs = []        # (round, step, ref_top2_margin, pre_div_noise)

    for rd in range(rounds):
        prev = _counts(ld)
        t = time.time()
        with torch.no_grad():
            out = model.generate(
                ids, attention_mask=attn, max_new_tokens=tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
                output_scores=True, return_dict_in_generate=True)
        dt = time.time() - t
        # scores: tuple length=tokens of [1, vocab]; stack → [T, vocab] fp32
        scores = torch.stack([s[0].float() for s in out.scores], dim=0)
        toks = out.sequences[0, -tokens:].tolist()

        d, s, h, m, f, e = _counts(ld)
        dd, ds, dh, dm, df, de = (d-prev[0], s-prev[1], h-prev[2],
                                  m-prev[3], f-prev[4], e-prev[5])

        if ref_scores is None:
            ref_scores = scores
            det = "REF"
        else:
            ref_arg = ref_scores.argmax(dim=1)
            cur_arg = scores.argmax(dim=1)
            # Walk steps in order: accumulate FP noise ONLY while argmax matches;
            # stop at the first flip (downstream is a different sequence).
            pre_div = 0.0
            div_step = None
            for st in range(scores.size(0)):
                if ref_arg[st].item() == cur_arg[st].item():
                    d = (scores[st] - ref_scores[st]).abs().max().item()
                    pre_div = max(pre_div, d)
                else:
                    div_step = st
                    break
            noise_floor = max(noise_floor, pre_div)
            if div_step is None:
                det = f"OK(noise={pre_div:.3f})"
            else:
                top2 = ref_scores[div_step].topk(2).values
                margin = (top2[0] - top2[1]).item()
                diverge_recs.append((rd, div_step, margin, pre_div))
                det = (f"FLIP(step={div_step} margin={margin:.3f} "
                       f"noise={pre_div:.3f})")

        if df != dm:
            print(f"[rank{rank}] round{rd} *** fetch({df}) != miss({dm}) ***", flush=True)
        print(f"[rank{rank}] round{rd} {dt:.1f}s det={det} "
              f"direct+={dd} staging+={ds} hit+={dh} miss+={dm} fetch+={df} "
              f"evict+={de} toks={toks}", flush=True)
        barrier()

    # ---- collective, MARGIN-AWARE verdict (all ranks ran all rounds) --------
    # A greedy token flip is benign iff it happened at a near-tie: the lead it
    # overturned (ref top-2 margin) is within the FP noise we actually observed
    # on non-flipping steps.  A flip at a margin >> noise_floor is NOT explained
    # by FP jitter → that is the corruption signature.  The post-flip cascade
    # logit diff (large) is an EFFECT of the flip, not evidence of corruption,
    # so it is deliberately excluded.
    import torch.distributed as dist
    nf = torch.tensor([noise_floor], device="cuda")
    if dist.is_initialized():
        dist.all_reduce(nf, op=dist.ReduceOp.MAX)
    g_noise = nf.item()
    thresh = max(2.0 * g_noise, 0.05)   # a flip below this margin = near-tie

    susp = [r for r in diverge_recs if r[2] > thresh]
    benign = [r for r in diverge_recs if r[2] <= thresh]
    for (rd_, st_, m_, n_) in susp:
        print(f"[rank{rank}] *** SUSPICIOUS FLIP round{rd_} step{st_} "
              f"margin={m_:.3f} > thresh={thresh:.3f} (noise_floor={n_:.3f}) "
              f"— argmax flipped a CLEAR winner, not a near-tie ***", flush=True)

    agg = torch.tensor([float(len(susp)), float(len(benign))], device="cuda")
    if dist.is_initialized():
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
    g_susp, g_benign = int(agg[0].item()), int(agg[1].item())

    if r0:
        print(f"[stress] SUMMARY rounds={rounds} tokens={tokens} "
              f"noise_floor(pre-flip |Δlogit|, max over ranks)={g_noise:.4f} "
              f"near_tie_flips={g_benign} suspicious_flips={g_susp} "
              f"flip_margin_thresh={thresh:.3f}", flush=True)
        if g_susp > 0:
            verdict = ("FAIL-SUSPICIOUS — argmax flipped where the winner led by "
                       f">{thresh:.3f}, beyond observed FP noise ⇒ investigate "
                       "fetch/staging data path")
        elif g_benign > 0:
            verdict = (f"PASS-BENIGN-FP — token flips only at near-ties "
                       f"(margin ≤ {thresh:.3f}, within cuBLAS/bf16 noise "
                       f"floor {g_noise:.3f}); cache/stream path adds no error")
        else:
            verdict = (f"PASS-DETERMINISTIC — zero token flips; "
                       f"pre-flip logit noise ≤ {g_noise:.4f}")
        print(f"[stress] VERDICT: {verdict}", flush=True)
        if ref_scores is not None:
            print(f"[rank0] canary decoded(round0)="
                  f"{tok.decode(out.sequences[0])[:140]!r}", flush=True)

    torch.cuda.synchronize()
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    ilog.proc_mem("pre_exit(stress)")
    os._exit(0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException as e:
        import traceback
        traceback.print_exc()
        print(f"[stress] FATAL {e!r}", flush=True)
        os._exit(3)
