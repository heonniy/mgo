"""Add untimed native pinned-cache telemetry to the private Infinity checkout."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/data2/esjung/tools/MoE-Infinity')
DEST=Path(__file__).resolve().parents[1]/'experiments/pcie_topology_ablation_20261009/baseline_source_patches'


def replace(path, old, new, expected=1):
    text=path.read_text()
    if new in text:
        return
    assert text.count(old)==expected, (str(path),text.count(old),expected)
    path.write_text(text.replace(old,new))


def main():
    header=ROOT/'core/memory/host_caching_allocator.h'
    implementation=header.with_suffix('.cpp')
    binding=ROOT/'core/python/py_archer_prefetch.cpp'
    replace(header,'#include <mutex>','#include <mutex>\n#include <utility>')
    replace(header,'  static std::mutex mutex_;','  static std::mutex mutex_;\n  static size_t allocated_bytes_;\n  static size_t peak_allocated_bytes_;')
    replace(header,'  static void record_free(void* ptr);','  static void record_free(void* ptr);\n  // Actual cudaHostAlloc payload, including cached blocks; one untimed snapshot.\n  static std::pair<size_t, size_t> pinned_memory_usage();')
    replace(implementation,'#include "host_caching_allocator.h"','#include "host_caching_allocator.h"\n#include <algorithm>')
    replace(implementation,'std::mutex HostCachingAllocator::mutex_;',
        'std::mutex HostCachingAllocator::mutex_;\nsize_t HostCachingAllocator::allocated_bytes_ = 0;\nsize_t HostCachingAllocator::peak_allocated_bytes_ = 0;\n\nstd::pair<size_t, size_t> HostCachingAllocator::pinned_memory_usage() {\n  std::lock_guard<std::mutex> guard(mutex_);\n  return {allocated_bytes_, peak_allocated_bytes_};\n}')
    replace(implementation,'  allocation_map_[ptr] = bytes;\n  return ptr;',
        '  allocation_map_[ptr] = bytes;\n  allocated_bytes_ += bytes;\n  peak_allocated_bytes_ = std::max(peak_allocated_bytes_, allocated_bytes_);\n  return ptr;')
    replace(implementation,'    allocation_map_.erase(it);','    allocated_bytes_ -= it->second;\n    allocation_map_.erase(it);')
    replace(implementation,'      allocation_map_.erase(ptr);','      allocated_bytes_ -= allocation_map_.at(ptr);\n      allocation_map_.erase(ptr);')
    replace(binding,'#include "kernel/ops.h"','#include "kernel/ops.h"\n#include "memory/host_caching_allocator.h"')
    old='      .def("get_expert_policy_stats",\n           &ArcherPrefetchHandle::GetExpertPolicyStats)'
    new=old+'\n      .def("get_host_pinned_memory_stats", [](const ArcherPrefetchHandle&) {\n        const auto usage = c10::HostCachingAllocator::HostCachingAllocator::pinned_memory_usage();\n        py::dict result;\n        result["allocated_bytes"] = usage.first;\n        result["peak_allocated_bytes"] = usage.second;\n        return result;\n      })'
    replace(binding,old,new,expected=2)
    DEST.mkdir(parents=True,exist_ok=True)
    patch=subprocess.check_output(['git','diff','--','core/memory/host_caching_allocator.h','core/memory/host_caching_allocator.cpp','core/python/py_archer_prefetch.cpp'],cwd=ROOT)
    (DEST/'infinity_native_pinned_memory.patch').write_bytes(patch)
    receipt=dict(status='PREPARED_NOT_COMPILED',checkout=str(ROOT),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (header,implementation,binding)},
        scope='Successful native HostCachingAllocator cudaHostAlloc payload including free cached blocks; excludes unrelated PyTorch pinned buffers and separate native staging pools',
        timing='O(1) counters under existing allocator lock; snapshot only after generation timing endpoints; allocation/admission/eviction/routing unchanged',
        verification='Private native build and physical GPU warmup still required; no compiled or measured result claimed')
    (DEST/'infinity_native_pinned_memory.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
