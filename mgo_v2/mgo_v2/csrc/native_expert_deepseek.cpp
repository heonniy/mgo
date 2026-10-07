#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/ops/index.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <vector>

at::Tensor silu_product_cuda(const at::Tensor&, const at::Tensor&);

// One Python/C++ crossing per ready wave. Per-expert GEMMs remain separate:
// this isolates host execution migration from grouped-GEMM scheduling.
std::vector<at::Tensor> execute_wave(
    const at::Tensor& cache, const at::Tensor& input,
    const at::Tensor& routing_weights, const std::vector<int64_t>& slots,
    const std::vector<at::Tensor>& rows, const std::vector<at::Tensor>& cols) {
  TORCH_CHECK(cache.is_cuda() && input.is_cuda() && routing_weights.is_cuda());
  TORCH_CHECK(cache.scalar_type() == at::kBFloat16 && input.scalar_type() == at::kBFloat16);
  TORCH_CHECK(routing_weights.scalar_type() == at::kBFloat16);
  TORCH_CHECK(cache.dim() == 2 && cache.size(1) == 8650752 && cache.is_contiguous());
  TORCH_CHECK(input.dim() == 2 && input.size(1) == 2048);
  TORCH_CHECK(routing_weights.dim() == 2 && routing_weights.size(0) == input.size(0));
  TORCH_CHECK(cache.device() == input.device() && cache.device() == routing_weights.device());
  TORCH_CHECK(slots.size() == rows.size() && slots.size() == cols.size());
  c10::cuda::CUDAGuard device_guard(cache.device());
  at::NoGradGuard no_grad;
  std::vector<at::Tensor> outputs;
  outputs.reserve(slots.size());
  for (size_t i = 0; i < slots.size(); ++i) {
    TORCH_CHECK(slots[i] >= 0 && slots[i] < cache.size(0));
    TORCH_CHECK(rows[i].device() == cache.device() && cols[i].device() == cache.device());
    TORCH_CHECK(rows[i].scalar_type() == at::kLong && cols[i].scalar_type() == at::kLong);
    TORCH_CHECK(rows[i].dim() == 1 && cols[i].sizes() == rows[i].sizes());
    auto w = cache.select(0, slots[i]);
    auto gate = w.narrow(0, 0, 2883584).view({1408, 2048});
    auto up = w.narrow(0, 2883584, 2883584).view({1408, 2048});
    auto down = w.narrow(0, 5767168, 2883584).view({2048, 1408});
    auto x = input.index_select(0, rows[i]);
    auto g = at::mm(x, gate.t());
    auto u = at::mm(x, up.t());
    auto activated = silu_product_cuda(g, u);
    auto y = at::mm(activated, down.t());
    auto weights = at::index(routing_weights, {rows[i], cols[i]}).unsqueeze(1);
    outputs.push_back(y * weights);
  }
  return outputs;
}

// Replace one NumPy full-slot search per active expert with one native scan.
// The arena owns the mapping; this routine only resolves it for this layer.
std::vector<int64_t> bind_layer_slots(
    pybind11::array_t<int32_t, pybind11::array::c_style | pybind11::array::forcecast> keys,
    pybind11::array_t<int32_t, pybind11::array::c_style | pybind11::array::forcecast> physical,
    int64_t layer, const std::vector<int64_t>& experts) {
  auto k = keys.unchecked<1>();
  auto p = physical.unchecked<1>();
  TORCH_CHECK(k.shape(0) >= p.shape(0), "physical slots exceed logical keys");
  TORCH_CHECK(layer >= 0 && layer < 26, "invalid layer");
  int64_t lookup[64];
  std::fill(std::begin(lookup), std::end(lookup), -1);
  for (pybind11::ssize_t i = 0; i < p.shape(0); ++i) {
    int64_t key = k(i);
    if (key >= layer * 64 && key < (layer + 1) * 64) {
      int64_t expert = key - layer * 64;
      TORCH_CHECK(lookup[expert] < 0, "duplicate main expert key");
      lookup[expert] = p(i);
    }
  }
  std::vector<int64_t> result;
  result.reserve(experts.size());
  for (auto expert : experts) {
    TORCH_CHECK(expert >= 0 && expert < 64 && lookup[expert] >= 0,
                "group expert missing from MAIN roles");
    result.push_back(lookup[expert]);
  }
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("execute_wave", &execute_wave, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("bind_layer_slots", &bind_layer_slots);
}
