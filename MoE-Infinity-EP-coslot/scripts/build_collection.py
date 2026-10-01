"""Build a per-request [L,E] history collection for OURS retrieval hotness.

Runs ShareGPT samples [OFFSET : OFFSET+N] (DISJOINT from the eval set [0:64]) with
batch_per_rank=1 (so each rank's accumulated demand == ONE request's [L,E]),
in=48/out=64 (same regime as eval), and saves [N, L, E] to --out.

Determinism not required here (offline prior); we just record routing.
"""
import argparse
import json
import os

from moe_infinity_ep.launch.distributed_setup import pin_visible_device
pin_visible_device()

from moe_infinity_ep.launch.distributed_setup import init_distributed, barrier  # noqa: E402
from moe_infinity_ep.launch.entry import MoE_EP  # noqa: E402
from moe_infinity_ep.utils.config import Config  # noqa: E402

INPUT_LEN = 48
OUTPUT_LEN = 64


def _sample_prompts(ds, tokenizer, seed, n_total):
    """IDENTICAL to coslot_owner_bench._sample_prompts (so eval[0:64] and
    collection[64:96] are disjoint by index under the SAME ordering)."""
    import random
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    sample_ids, texts, tok_rows = [], [], []
    for i in idxs:
        row = ds[i]
        convs = row["conversations"]
        human = next((c["value"] for c in convs
                      if c.get("from") == "human" and c.get("value")), None)
        if not human:
            continue
        ids = tokenizer(human, add_special_tokens=True).input_ids
        if len(ids) < INPUT_LEN:
            continue
        sample_ids.append(row["id"])
        texts.append(human)
        tok_rows.append(ids[:INPUT_LEN])
        if len(sample_ids) == n_total:
            break
    if len(sample_ids) < n_total:
        raise RuntimeError(f"only {len(sample_ids)}/{n_total} prompts >= {INPUT_LEN} tok")
    return sample_ids, texts, tok_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=32, dest="n")   # collection size
    ap.add_argument("--offset", type=int, default=64)   # start AFTER eval [0:64]
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.environ["MOE_EP_OURS_EVICT_COST"] = "retrieval"   # -> controller builds rank_accum
    cfg = Config.from_yaml(args.config)
    world = int(os.environ["WORLD_SIZE"])
    cfg.resolve_parallel(world)
    topology = init_distributed(cfg.parallel.ep_size)
    rank = topology.global_rank

    import numpy as np
    import torch
    import torch.distributed as dist
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path, trust_remote_code=True)
    arrow = ("/home/work/hyewon.lee/dataset/Aeala___share_gpt_vicuna_unfiltered/"
             "default/0.0.0/8b0048ad6ae8c22f46a78c15559dec98feef5539/"
             "share_gpt_vicuna_unfiltered-train.arrow")
    ds = load_dataset("arrow", data_files=arrow, split="train")
    # need offset+n qualifying samples; take the [offset:offset+n] slice (DISJOINT
    # from eval [0:64] as long as offset>=64).
    n_need = args.offset + args.n
    all_ids, _texts, all_rows = _sample_prompts(ds, tokenizer, args.seed, n_need)
    col_ids = all_ids[args.offset:args.offset + args.n]
    col_rows = all_rows[args.offset:args.offset + args.n]
    assert len(col_rows) == args.n, f"only {len(col_rows)} samples (need {args.n})"
    if topology.is_rank0:
        print(f"[build] collection n={args.n} offset={args.offset} "
              f"(eval=[0:64] disjoint) world={world}", flush=True)

    engine = MoE_EP(cfg, topology)
    engine.load()
    model = engine.model; model.eval()
    exec_ = engine.engine.expert_executor
    ctrl = exec_.controller
    assert getattr(ctrl, "_rank_accum", None) is not None, "rank_accum not wired"

    def sync():
        torch.cuda.synchronize(); barrier()

    n_batches = (args.n + world - 1) // world
    entries = []  # rank0 collects [L,E] per request
    for b in range(n_batches):
        lo = b * world
        # this rank's single sample for this batch
        my = lo + rank
        if my < args.n:
            ids = torch.tensor([col_rows[my]], dtype=torch.long).cuda()  # [1,48]
        else:
            ids = torch.tensor([col_rows[0]], dtype=torch.long).cuda()   # pad (ignored)
        attn = torch.ones_like(ids)
        ctrl.reset_rank_accum()
        sync()
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            cur_attn = attn
            for _ in range(OUTPUT_LEN):
                cur_attn = torch.cat(
                    [cur_attn, torch.ones(cur_attn.size(0), 1, dtype=cur_attn.dtype,
                                          device=cur_attn.device)], dim=1)
                out = model(input_ids=nxt, attention_mask=cur_attn,
                            past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        sync()
        # rank_accum[G,L,E] is identical on every rank (all-gathered) -> rank0
        # reads each rank's [L,E] = that rank's single request.
        if topology.is_rank0:
            for g in range(world):
                idx = lo + g
                if idx < args.n:
                    entries.append(ctrl._rank_accum.rank_matrix(g))
            print(f"[build] batch {b+1}/{n_batches} captured "
                  f"(total {len(entries)})", flush=True)

    if topology.is_rank0:
        arr = np.stack(entries, axis=0)  # [N, L, E]
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.save(args.out, arr)
        meta = {"n": int(arr.shape[0]), "L": int(arr.shape[1]), "E": int(arr.shape[2]),
                "offset": args.offset, "col_ids": col_ids, "seed": args.seed}
        with open(args.out + ".meta.json", "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[build] saved {arr.shape} -> {args.out}", flush=True)

    barrier()
    torch.cuda.synchronize()
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
