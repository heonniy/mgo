"""Small, isolated DeepSeek expert H2D and GEMM service-time probe.

Run through run_headline_job.py so the four owner inference loads are stopped
and restored. Only CUDA device 0 is used. This is not a full-model timing run.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch


def timed_events(fn, warmup=10, repeats=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    pairs = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        pairs.append((start, end))
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in pairs]


def summarize(samples):
    ordered = sorted(samples)
    return {
        'median_ms': statistics.median(ordered),
        'p05_ms': ordered[max(0, int(len(ordered) * .05) - 1)],
        'p95_ms': ordered[min(len(ordered) - 1, int(len(ordered) * .95))],
        'samples': len(ordered),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.manual_seed(417)

    # DeepSeek-V2-Lite routed expert: gate/up [1408,2048], down [2048,1408],
    # BF16, exactly 8,650,752 elements = 16.5 MiB.
    host = torch.randn(8650752, dtype=torch.bfloat16).pin_memory()
    weight = torch.empty_like(host, device='cuda')
    copy_stream = torch.cuda.Stream()
    with torch.cuda.stream(copy_stream):
        copy_ms = timed_events(lambda: weight.copy_(host, non_blocking=True))
    copy_stream.synchronize()

    gate = weight[:2883584].view(1408, 2048)
    up = weight[2883584:5767168].view(1408, 2048)
    down = weight[5767168:].view(2048, 1408)
    cases = {}
    for rows in (1, 3, 5, 8, 17):
        x = torch.randn(rows, 2048, dtype=torch.bfloat16, device='cuda')
        g = torch.empty(rows, 1408, dtype=torch.bfloat16, device='cuda')
        u = torch.empty_like(g)
        a = torch.randn_like(g)
        y = torch.empty(rows, 2048, dtype=torch.bfloat16, device='cuda')

        def three_gemms():
            torch.mm(x, gate.t(), out=g)
            torch.mm(x, up.t(), out=u)
            # Preallocated activation isolates the GEMMs, not SiLU work.
            torch.mm(a, down.t(), out=y)

        case = {'three_gemm_cuda_interval': summarize(timed_events(three_gemms))}
        for key, matrix, output in (
            ('gate_gemm', gate, g), ('up_gemm', up, u), ('down_gemm', down, y)):
            source = a if key == 'down_gemm' else x
            case[key] = summarize(timed_events(
                lambda source=source, matrix=matrix, output=output:
                torch.mm(source, matrix.t(), out=output)))
        cases[str(rows)] = case

    result = {
        'status': 'PASS',
        'scope': 'Standalone one-GPU synthetic DeepSeek expert, no model/dispatch/comm',
        'physical_gpu': 0,
        'expert_bytes': host.numel() * host.element_size(),
        'copy_cuda_interval': summarize(copy_ms),
        'cases': cases,
        'interpretation': 'CUDA event intervals include submission gaps; three-GEMM interval excludes activation, gather, weighting, and scheduler.',
    }
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
