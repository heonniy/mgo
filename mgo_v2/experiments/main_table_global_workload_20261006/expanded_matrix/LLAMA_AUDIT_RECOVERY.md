# Placement-audit parser recovery

The initial16-thread audit failed before warmup or primary timing: the parser
recognized only `buffer type overridden to CPU`, while this installed llama
loader reports `CUDA_Host` for CPU-directed expert overrides. The loader source
selects from buft_list_cpu for CPU overrides and later reports mmap fallback
from CUDA_Host to CPU (102 experts plus embedding). This is a host-buffer label
compatibility issue, not evidence that experts were silently placed on GPU.

Accept only exact CPU or CUDA_Host names with line-end boundary; CUDA0 etc.
remain rejected. Preserve exact102 tensors and3 tensors/layer for layers0..33,
all144 accounting, op_offload=false, token-parity and timing recomputation gates.
Recorded failure log matches102 unique expert overrides in exactly34 layers.
Rebuild with fresh source/binary hashes. Run under new v2 labels, retaining
v1 failure. No model, thread count, placement or timing-method change.
