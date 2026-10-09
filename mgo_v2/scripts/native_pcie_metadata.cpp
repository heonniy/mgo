#include <algorithm>
#include <cstdint>
#include <cstddef>

// Preserve GateHistory's add-then-remove FP64 accumulation order exactly.
extern "C" int mgo_gate_update(double* ring, double* sums, int64_t* heads,
    int64_t* counts, const double* probs, int layers, int experts, int window,
    int layer, int tokens) {
  if (layer < 0 || layer >= layers || tokens < 0 || experts < 1 || window < 1)
    return -1;
  double* total = sums + static_cast<size_t>(layer) * experts;
  double* history = ring + static_cast<size_t>(layer) * window * experts;
  if (tokens >= window) {
    std::fill(total, total + experts, 0.0);
    const double* latest = probs + static_cast<size_t>(tokens-window)*experts;
    for (int row = 0; row < window; ++row)
      for (int e = 0; e < experts; ++e) {
        double value = latest[static_cast<size_t>(row)*experts+e];
        history[static_cast<size_t>(row)*experts+e] = value;
        total[e] += value;
      }
    heads[layer] = 0; counts[layer] = window;
    return 0;
  }
  for (int row = 0; row < tokens; ++row) {
    const bool full = counts[layer] == window;
    const int position = full ? heads[layer] : (heads[layer]+counts[layer])%window;
    double* previous = history + static_cast<size_t>(position)*experts;
    for (int e = 0; e < experts; ++e) {
      double value = probs[static_cast<size_t>(row)*experts+e];
      total[e] += value;
      if (full) total[e] -= previous[e];
      previous[e] = value;
    }
    if (full) heads[layer] = (heads[layer]+1)%window;
    else ++counts[layer];
  }
  return 0;
}

extern "C" void mgo_gate_scores(const double* sums, const int64_t* counts,
    float* scores, int experts, int layer) {
  const double divisor = static_cast<double>(std::max<int64_t>(1,counts[layer]));
  for (int e = 0; e < experts; ++e)
    scores[static_cast<size_t>(layer)*experts+e] =
        static_cast<float>(sums[static_cast<size_t>(layer)*experts+e]/divisor);
}

extern "C" int mgo_route_histogram(const uint8_t* ids, int64_t* hist,
    int world, int batch, int topk, int experts) {
  std::fill(hist,hist+world*experts,0);
  for (int r=0;r<world;++r)
    for (int row=0;row<batch;++row)
      for (int k=0;k<topk;++k) {
        int expert=ids[(r*batch+row)*topk+k];
        if (expert>=experts) return -1;
        ++hist[r*experts+expert];
      }
  return 0;
}
