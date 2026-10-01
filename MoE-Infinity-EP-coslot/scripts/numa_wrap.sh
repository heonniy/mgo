#!/usr/bin/env bash
# Per-rank wrapper that binds the worker to its GPU's NUMA node before exec.
# Invoked by torchrun (one instance per nproc_per_node).
#
# Required env (torchrun provides): LOCAL_RANK
# Optional: GPU_NUMA_MAP="0:0,1:1" — comma-separated LOCAL_RANK:NUMA pairs.
# Default heuristic if unset: query /sys/bus/pci for the GPU's NUMA node.
set -euo pipefail

local_rank="${LOCAL_RANK:?LOCAL_RANK must be set (torchrun)}"

resolve_numa() {
    # If user provided an explicit map, honor it.
    if [[ -n "${GPU_NUMA_MAP:-}" ]]; then
        for kv in ${GPU_NUMA_MAP//,/ }; do
            local k="${kv%%:*}" v="${kv##*:}"
            if [[ "$k" == "$local_rank" ]]; then echo "$v"; return; fi
        done
    fi
    # Otherwise query GPU<->NUMA mapping via nvidia-smi + sysfs.
    # CUDA_VISIBLE_DEVICES gating happens AFTER this script (inside Python),
    # so nvidia-smi still sees all devices here.
    local bus
    bus="$(nvidia-smi --query-gpu=pci.bus_id -i "$local_rank" --format=csv,noheader 2>/dev/null | tr 'A-Z' 'a-z' | sed 's/^0*//;s/^/0000/')"
    local numa_file="/sys/bus/pci/devices/${bus}/numa_node"
    if [[ -r "$numa_file" ]]; then
        local n
        n="$(cat "$numa_file")"
        if [[ "$n" -ge 0 ]]; then echo "$n"; return; fi
    fi
    echo "0"  # safe fallback
}

numa_node="$(resolve_numa)"

echo "[numa_wrap] LOCAL_RANK=$local_rank -> NUMA $numa_node" >&2

# Surface NCCL diagnostics on stderr (still WARN-level so noise is bounded).
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Disable IPv6 so init_method=env:// doesn't try unreachable addresses first
# (we saw "Address family not supported by protocol" warnings).
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

# --- M7 NCCL tuning (intra-node, NVLink) ---
# Bigger network buffers + more threads help all-to-all throughput for many
# small chunks. Defaults err on the conservative side; override via env.
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-8388608}"        # 8 MB
export NCCL_NTHREADS="${NCCL_NTHREADS:-256}"
# Prefer NVLink over PCIe for peer-to-peer.
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"

exec numactl --cpunodebind="$numa_node" --membind="$numa_node" \
    python "$@"
