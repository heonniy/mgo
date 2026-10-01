# MoE-Infinity intra-node multi-GPU execution flow

This note describes the current code path in:

- `/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity`
- Focus: Qwen3 MoE block, original MoE-Infinity runtime, multiple visible GPUs in one node.

Important terminology:

- In the original intra-node path, execution is mostly single-process multi-GPU. The code chooses a CUDA `gpu_id`, not a distributed process rank.
- The RPC-style `rank` path exists in `moe_infinity/distributed/expert_executor.py::dispatch`, but Qwen3 currently calls `dispatch_local()`, so the practical intra-node path is GPU-index based.
- When this document says "fetch rank" for original MoE-Infinity, read it as "target GPU ID inside the process". In `MoE-Infinity-EP`, this is changed into real torchrun ranks.

## Source map

Core files:

- `moe_infinity/models/qwen.py`
  - Qwen3 sparse MoE forward.
  - Builds router results, calls current-layer cache protection, predictor prefetch, then local expert dispatch.
- `moe_infinity/distributed/expert_executor.py`
  - Python side dedup and target GPU selection for on-demand execution.
- `moe_infinity/memory/expert_prefetcher.py`
  - Python side current-layer candidate protection and future prefetch enqueue.
- `core/parallel/expert_dispatcher.cpp`
  - C++ expert cache hit/miss handling, on-demand fetch, execution, output combine.
- `core/prefetch/archer_prefetch_handle.cpp`
  - C++ binding for prefetch enqueue and cache-candidate replacement.
- `core/prefetch/task_scheduler.cpp`
  - Background prefetch task queue and eviction before prefetch.
- `core/model/model_topology.cpp`
  - Model topology, sparse/dense node default device placement, sparse cache limit.

## Setup-time placement

During model load, `OffloadEngine.setup_archer_hooks()` registers tensors and builds a model topology.

The topology groups parameters into nodes:

- Dense nodes: normal non-expert parameters.
- Sparse nodes: each expert's tensor group.

Then `ArcherTopologyHandle::InitializeTopology()` places nodes:

1. Dense nodes are moved to GPUs and split roughly across visible GPUs.
2. Sparse expert nodes are initially moved to CPU.
3. Each sparse node still gets a `default_device`, assigned round-robin across visible GPUs.

This `default_device` matters for prefetch. It is not necessarily identical to `expert_id % num_gpus`, because the round-robin cursor continues after dense placement.

The C++ `ExpertDispatcher` is also constructed with one fetch queue and one or more exec queues per visible CUDA device. It uses `kNumDevices()` / `torch.cuda.device_count()` style visibility, so if the process sees 4 GPUs, it creates per-GPU machinery for 4 GPUs.


## GPU and CPU memory model

Original MoE-Infinity's intra-node memory model is best understood as:

```text
Disk/offload files
  -> CPU host memory backing store
  -> GPU dense placement + sparse expert cache
```

At runtime there is one global C++ `ArcherPrefetchHandle` per process. That handle initializes:

- `DeviceMemoryPool`: one logical memory budget per visible CUDA device.
- `HostMemoryPool`: one shared CPU host memory pool for the process.
- `ArcherTopologyHandle`: the graph of dense nodes and sparse expert nodes.
- `ArcherTaskPool`: background prefetch workers.

### GPU memory budget

The Python config has `device_memory_ratio`. `OffloadEngine.init()` passes only this value into the C++ `prefetch_handle`.

In C++:

```cpp
kDeviceMemoryPool->SetMemoryRatio(device_memory_ratio);
```

So the effective Archer budget per GPU is:

```text
gpu_budget[g] = physical_gpu_memory[g] * device_memory_ratio
```

After topology initialization:

```text
sparse_cache_budget[g] = gpu_budget[g] - dense_nodes_placed_on_gpu[g]
```

Dense nodes are placed on GPUs during setup. Sparse expert nodes are initially moved to CPU, but each sparse node is assigned a `default_device` round-robin across visible GPUs.

The expert dispatcher stores this remaining sparse capacity in `cache_sizes_[gpu_id]`. When a miss expert is fetched to a GPU, the expert's byte size is subtracted from that GPU's sparse cache. When an expert is evicted, its size is added back.

### CPU memory budget

The Python config also has `host_memory_ratio`, but in the current original MoE-Infinity code path it is not passed into the C++ `prefetch_handle` constructor. The C++ `HostMemoryPool` uses a compile-time macro:

```cpp
HOST_MEMORY_RATIO  // default 0.8
```

and initializes:

```text
host_capacity = total_system_memory * HOST_MEMORY_RATIO
```

There is an important caveat: `HostMemoryPool::AllocateMemory()` allocates via PyTorch's host caching allocator, but the current implementation does not subtract from `free_memory_` or enforce the capacity on allocation. So practically, CPU memory pressure is governed more by real available RAM/cgroup limits than by the Python `host_memory_ratio` setting.

Conceptually, CPU memory is the backing store for offloaded parameters. Sparse expert nodes are moved to CPU at setup, then copied into GPU sparse cache on demand or by prefetch. When an expert is evicted from GPU, it goes back to its default host device, normally CPU.

## Four-GPU memory example

Assume:

- 4 visible GPUs: `cuda:0..cuda:3`
- Each GPU has 80 GB physical memory.
- `device_memory_ratio = 0.75`
- Total dense weights placed on each GPU after topology setup are approximately:

```text
cuda:0 dense = 12 GB
cuda:1 dense = 12 GB
cuda:2 dense = 12 GB
cuda:3 dense = 10 GB
```

Then Archer's GPU memory budget is:

```text
80 GB * 0.75 = 60 GB per GPU
```

Sparse expert cache budget becomes:

```text
cuda:0 sparse cache ~= 60 - 12 = 48 GB
cuda:1 sparse cache ~= 60 - 12 = 48 GB
cuda:2 sparse cache ~= 60 - 12 = 48 GB
cuda:3 sparse cache ~= 60 - 10 = 50 GB
```

If one expert is about 9.4 MB, the rough cache slot count is:

```text
cuda:0 ~= 48 GB / 9.4 MB ~= 5200 experts
cuda:1 ~= 5200 experts
cuda:2 ~= 5200 experts
cuda:3 ~= 5400 experts
```

This is only a sizing intuition. The real code tracks bytes, not a fixed slot count, and expert sizes can vary.

With 4 GPUs, current-layer on-demand miss target still uses:

```text
target_gpu = expert_id % 4
```

So:

```text
expert 0 -> cuda:0
expert 1 -> cuda:1
expert 2 -> cuda:2
expert 3 -> cuda:3
expert 4 -> cuda:0
...
```

But prefetch target uses the expert node's `default_device`, assigned during topology setup, not necessarily `expert_id % 4`.


## Per-layer Qwen3 flow

At each Qwen3 MoE layer:

1. `Qwen3MoEBlock.forward()` flattens hidden states from `[batch, seq, hidden]` to `[tokens, hidden]`.
2. It computes:
   - `router_logits`
   - `router_mask`: boolean `[tokens, num_experts]`
   - `routing_weights_mask`: weights `[tokens, num_experts]`
3. It separately computes `selected_experts = topk(router_logits)`, reshaped to `expert_index = [batch, seq, top_k]`.
4. It calls:
   - `expert_prefetcher.fetch_experts_lock_cache(layer_id, expert_index)`
   - `expert_predictor.predict(...)`
   - `expert_prefetcher.prefetch_experts(layer_id, expert_matrix)`
   - `expert_executor.dispatch_local(...)`
   - `expert_executor.wait_dispatch_local()`

The actual required expert set for this layer is computed in `dispatch_local()`, not in the prefetcher.

## Current-layer expert dedup

`DistributedExpertExecutor.dispatch_local()` computes:

```python
expert_count = torch.sum(router_mask.view((-1, num_expert)), dim=0)
expert_list = np.arange(num_expert)[expert_count > 0].tolist()
```

This deduplicates by expert ID. If 100 tokens route to expert 17, expert 17 is enqueued once for that layer. The C++ dispatcher then handles all token rows for expert 17 in one execution.

The list is effectively sorted by expert ID because it comes from `np.arange(num_expert)`.

At this Python stage, the code does not yet know how many of these experts are cache hits or misses. Hit/miss is decided in C++ per expert node.

## On-demand target GPU selection

For every expert in the deduped `expert_list`, `dispatch_local()` chooses:

```python
gpu_id = expert_id % torch.cuda.device_count()
expert_dispatcher.enqueue_expert(layer_id, expert_id, gpu_id, False)
```

So the initial on-demand fetch target is a static modulo rule.

Properties:

- It does not use token count. A hot expert with many routed tokens is not specially load-balanced.
- It does not use cache pressure.
- It does not consult `node.default_device`.
- It only uses the expert ID and the number of visible GPUs.

But this modulo target can be overridden if the expert is already cached.

## C++ enqueue: hit or miss

`ExpertDispatcher::Enqueue()` receives `(layer_idx, expert_idx, gpu_id)`.

It looks up the expert node and checks:

```cpp
bool cache_hit = expert_node->node->device.is_cuda();
```

If it is a cache hit:

1. The target GPU is overwritten:

   ```cpp
   args.gpu_id = expert_node->node->device.index();
   ```

2. The expert is sent directly to that GPU's exec queue.
3. No fetch is needed.

If it is a cache miss:

1. The request is pushed to `input_queue_[gpu_id]`.
2. That `gpu_id` is the modulo target chosen by Python.
3. The fetch thread for that GPU will later load the expert.

So the real execution target is:

- Hit: wherever the expert already lives.
- Miss: `expert_id % num_visible_gpus`.

## On-demand miss fetch

Each GPU has a `GPUFetchFunc(gpu_id)` thread. It pops miss requests from `input_queue_[gpu_id]`.

For a miss, it checks whether the target GPU has enough sparse cache space:

- If there is enough space:
  - It reserves cache space.
  - Adds `(layer_idx, expert_idx)` to `cached_experts_[gpu_id]`.
  - Calls `node->SetDevice(cuda:gpu_id, on_demand=true, stream)`.

- If there is not enough space:
  - For prefill-like execution (`batch_size > 1` and not decode), it allows one temporary overflow slot.
  - For decode or batch size 1, it evicts an existing cached expert.

The on-demand eviction policy in `ExpertDispatcher::FindExpertEvict()` chooses the cached expert with the smallest `incache_visit_count` among unlocked cached experts on that GPU. That is effectively LFU-ish.

After fetch, the dispatcher pushes an `ExecArgs` item to `exec_queue_[gpu_id]`.

## Expert execution and output combine

Each GPU also has `GPUExecFunc(gpu_id)`.

For each expert execution:

1. It builds `token_mask = router_mask[..., expert_idx]`.
2. In decode-like single-token mode, it sends the whole hidden state to that GPU.
3. In multi-token prefill mode, it selects only token rows for that expert.
4. It binds expert weights from the fetched/cached blob.
5. It runs the expert MLP on the GPU stream.
6. `OutputFunc()` moves the output back to the original output device and accumulates:
   - Decode: `final_hidden_states_.add_(output * weight)`
   - Prefill: `final_hidden_states_.index_add_(token_indices, weighted_output)`

When all queued experts finish, `wait_dispatch_local()` returns the accumulated hidden states.


## Batch and token constraints

There is no explicit Python-level maximum batch size in the Qwen3 wrapper. The effective hard limit is in C++ expert execution:

```cpp
static const int64_t kMaxTokens = 1024;
```

`MoEMLP::forward()` checks:

```text
0 < expert_input_tokens <= 1024
```

The variable is named `batch_size` in C++, but in this path it means the number of token rows sent to one expert invocation.

For prefill:

```text
total_tokens = batch_size * sequence_length
expert_input_tokens = number of token rows routed to this expert
```

Worst case, all tokens route to the same expert, so a safe conservative condition is:

```text
batch_size * sequence_length <= 1024
```

For decode with `sequence_length = 1`:

```text
expert_input_tokens <= batch_size
```

so a conservative condition is:

```text
batch_size <= 1024
```

In practice, if routing is spread across experts, `batch_size * sequence_length` can be larger than 1024 as long as no single expert receives more than 1024 token rows. The code does not chunk a too-large expert batch; it fatal-errors.

Batch shape also affects cache behavior:

- If `hidden_states_.size(0) > 1` and `is_decode == false`, the dispatcher treats it as prefill-like and allows one temporary GPU cache overflow slot when sparse cache is full.
- If it is decode or effectively one-token execution, it evicts an existing cached expert instead of using that overflow path.


## Prefetch path

There are two related prefetch-side calls in Qwen3 forward.

### 1. Current-layer cache-candidate protection

`fetch_experts_lock_cache(layer_id, expert_index)` maps current layer experts to tensor IDs, then calls:

```python
archer_engine.replace_cache_candidates(tensor_ids)
```

Despite the method name, this does not fetch the experts. It updates the task scheduler's `candidates_` set and clears lower-priority prefetch queues. Later, `RemoveCachedSparseNode()` skips nodes in `candidates_`, so these tensors are protected from prefetch eviction.

Caution in the current Qwen path:

- `expert_index` has shape `[batch, seq, top_k]`.
- `fetch_experts_lock_cache()` currently iterates it directly as if it were a flat list of expert IDs.
- For Qwen3, this looks like it should be flattened and deduplicated first. The dispatch path dedups correctly from `router_mask`; the cache-protection path is less robust as written.

### 2. Future expert prefetch

For each batch item, `ExpertPredictor.predict()` updates the trace and returns `expert_matrix`, a `[num_layers, num_experts]` score matrix.

`ExpertPrefetcher.prefetch_experts()` then:

1. Iterates layers from the current layer onward.
2. Collects `(tensor_id, score)` for every positive score.
3. Sorts by score descending.
4. Builds `tensor_ids`.
5. Asserts `tensor_ids` are unique.
6. Calls `replace_cache_candidates(tensor_ids)`.
7. For each tensor ID:

   ```python
   gpu_id = archer_engine.get_node_default_device([tensor_id])
   archer_engine.enqueue_prefetch(tensor_id, gpu_id)
   ```

In C++, however, `ArcherPrefetchHandle::EnqueuePrefetch()` currently ignores the `gpu_id` argument:

```cpp
// task->dst_device = CUDA_DEVICE(gpu_id); // use default device for now
task->dst_device = node->default_device;
```

So prefetch target GPU is the expert node's `default_device`, not necessarily the Python-computed `gpu_id`, and not necessarily `expert_id % num_gpus`.

The task is enqueued with:

- `priority = 1`
- `on_demand = false`
- `src_device = node->device`
- `dst_device = node->default_device`

On-demand tasks use higher priority paths, so prefetch is opportunistic.

## Prefetch worker and eviction

`ArcherTaskPool` has worker threads per GPU. Each worker scans the priority queues and picks a task whose `dst_device.index()` matches that worker's GPU.

For prefetch tasks:

1. It calls `RemoveCachedSparseNode(node)` to make room.
2. That function skips:
   - nodes in `candidates_`
   - nodes currently in the exec queue
   - locked nodes
3. Then it calls `SetNodeDevice(task)` to move the node to `task->dst_device`.

Note: the prefetch-side sparse eviction code sorts by `incache_visit_count` in descending order before trying evictions. That reads differently from the on-demand dispatcher, which chooses the smallest visit count. This is worth checking before relying on it as a clean LFU policy.

## Worked example

Assume:

- 2 visible GPUs: `cuda:0`, `cuda:1`
- Current layer: `L`
- `top_k = 2`
- Four local tokens route as:

```text
token0 -> experts [3, 9]
token1 -> experts [3, 17]
token2 -> experts [4, 17]
token3 -> experts [12, 13]
```

`dispatch_local()` collapses this into:

```text
expert_list = [3, 4, 9, 12, 13, 17]
```

So even though expert 3 appears twice and expert 17 appears twice, each expert is enqueued once.

Initial on-demand target GPU:

```text
expert 3  -> 3 % 2  = cuda:1
expert 4  -> 4 % 2  = cuda:0
expert 9  -> 9 % 2  = cuda:1
expert 12 -> 12 % 2 = cuda:0
expert 13 -> 13 % 2 = cuda:1
expert 17 -> 17 % 2 = cuda:1
```

Suppose current cache state is:

```text
expert 9 is already cached on cuda:0
expert 12 is already cached on cuda:0
all others are misses
```

Then the real execution targets become:

```text
expert 3  miss -> fetch to cuda:1
expert 4  miss -> fetch to cuda:0
expert 9  hit  -> execute on cuda:0, overriding modulo target cuda:1
expert 12 hit  -> execute on cuda:0
expert 13 miss -> fetch to cuda:1
expert 17 miss -> fetch to cuda:1
```

Miss count here is 4:

```text
miss experts = [3, 4, 13, 17]
```

Hit count is 2:

```text
hit experts = [9, 12]
```

If `cuda:1` has insufficient sparse cache space for expert 17 during decode, `FindExpertEvict(cuda:1)` chooses the cached expert on `cuda:1` with the smallest `incache_visit_count`, provided it can lock the node.

Now suppose the predictor returns future scores:

```text
(L+1, expert 2)  score 0.80
(L+2, expert 17) score 0.50
(L+1, expert 4)  score 0.30
```

The prefetcher sorts:

```text
[(L+1, 2), (L+2, 17), (L+1, 4)]
```

For each tensor ID it asks `get_node_default_device()`, then calls `enqueue_prefetch(tensor_id, gpu_id)`.

But the C++ prefetch implementation uses `node->default_device`, so the actual prefetch destination is:

```text
prefetch target = topology-assigned default_device for that expert node
```

not the on-demand modulo target.

## RPC rank path, for comparison

There is another method, `DistributedExpertExecutor.dispatch()`, that uses RPC and `DeviceMapManager.get_target_device()`.

That path:

- Computes a deduped `expert_list` similarly.
- Calls `device_map_manager.get_target_device(expert_list)`.
- That manager scatters experts across `(rank, gpu_id, expert_id)` tuples using `world_size * device_per_node`.
- It random-shuffles GPU IDs inside the loop.
- If `world_size > 1`, it starts from `base = 1`, which means rank 0 is skipped as a target in that code path.

However, Qwen3 forward calls `dispatch_local()`, not `dispatch()`. So for the practical Qwen3 intra-node path, the RPC rank assignment is not the active path.

## Contrast with MoE-Infinity-EP

Original MoE-Infinity intra-node multi-GPU:

- One process sees multiple GPUs.
- Target is a local `gpu_id`.
- Miss target uses `expert_id % num_visible_gpus`.
- Hit target is the GPU where the expert is already cached.
- Token activations are copied inside the process to the target CUDA device.
- No NCCL all-to-all in the active Qwen3 local path.

MoE-Infinity-EP:

- One torchrun process per GPU.
- Target is a real EP rank.
- It builds a global replicated cache view.
- It classifies hits/misses across ranks.
- It routes token activations with NCCL `all_to_all_single`.
- Fetch/placement target can be swapped by policy.

## Key takeaways

1. Current-layer expert demand is deduplicated in `dispatch_local()` by summing `router_mask`.
2. On-demand miss target GPU is `expert_id % torch.cuda.device_count()`.
3. Cache hits override the modulo target and execute where the expert already lives.
4. On-demand eviction in `ExpertDispatcher` is based on the smallest `incache_visit_count` among cached experts on the target GPU.
5. Prefetch target GPU is `node->default_device`, assigned by topology, because `EnqueuePrefetch()` ignores the Python `gpu_id` argument.
6. `replace_cache_candidates()` protects likely-needed nodes from prefetch eviction and clears older prefetch queues.
7. The active Qwen3 intra-node path is not true distributed rank-based EP. It is local multi-GPU dispatch inside one process.

