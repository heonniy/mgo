// Opt-in exact-only controller for the PCIe study. Routing aggregation,
// quotas, admission, Gate eviction, owner updates and accounting run in C++.
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

using I = std::int64_t;
constexpr int E = 128, L = 48, K = E * L, W = 4;
constexpr I INF = I{1} << 50;

struct State {
  std::int32_t *capacities, *slots;
  std::int16_t *owner;
  std::int8_t *primary;
  std::int32_t *last;
  std::uint8_t *seen, *lost;
  std::int32_t *birth, *reuses;
  float *gates;
  int stride;
};
struct Output {
  I *targets;
  std::int16_t *effective;
  double *masses;
  std::int8_t *lengths, *destinations;
  I *fetches, *quotas;
  double *row;
};
struct Engine {
  std::mt19937 random;
  int policy, quota_mode;
  bool weighted;
  std::array<I, W * W> costs;
  std::vector<std::array<double,61>> trace;
  Engine(int seed, int p, int q, const I *c)
    : random(seed), policy(p), quota_mode(q), weighted(c != nullptr) {
    if (c) std::copy(c, c + W * W, costs.begin());
    trace.reserve(4096);
  }
};

static void quota(int n, int mode, int event, I *q) {
  if (!mode) {
    for (int r = 0; r < W; ++r) q[r] = n / W + (r < n % W);
  } else {
    for (int g = 0; g < 2; ++g) {
      const int total = n / 2 + ((n % 2) && g == event % 2);
      const int priority = (event / 2 + 1) % 2;
      for (int r = 0; r < 2; ++r)
        q[g * 2 + r] = total / 2 + ((total % 2) && r == priority);
    }
  }
}

// Match the baseline Numba quicksort's equal-key order, including its
// median pivot and 15-element insertion-sort cutoff.
static std::vector<int> descending_order(const std::vector<I>& values) {
  std::vector<int> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  if (order.size() < 2) return order;
  std::vector<std::pair<int,int>> stack{{0, int(order.size()) - 1}};
  auto before = [&](int a, int b) { return values[a] > values[b]; };
  while (!stack.empty()) {
    auto [lo, hi] = stack.back(); stack.pop_back();
    while (hi - lo >= 15) {
      const int mid = (lo + hi) / 2;
      if (before(order[mid], order[lo])) std::swap(order[mid], order[lo]);
      if (before(order[hi], order[mid])) std::swap(order[hi], order[mid]);
      if (before(order[mid], order[lo])) std::swap(order[mid], order[lo]);
      const int pivot = order[mid]; std::swap(order[mid], order[hi]);
      int i = lo, j = hi - 1;
      for (;;) {
        while (i < hi && before(order[i], pivot)) ++i;
        while (j >= lo && before(pivot, order[j])) --j;
        if (i >= j) break;
        std::swap(order[i++], order[j--]);
      }
      std::swap(order[i], order[hi]);
      if (hi - i > i - lo) {
        if (hi > i) stack.emplace_back(i + 1, hi);
        hi = i - 1;
      } else {
        if (i > lo) stack.emplace_back(lo, i - 1);
        lo = i + 1;
      }
    }
    for (int i = lo + 1; i <= hi; ++i) {
      const int item = order[i]; int j = i;
      while (j > lo && before(item, order[j-1])) { order[j] = order[j-1]; --j; }
      order[j] = item;
    }
  }
  return order;
}

static std::vector<int> assign(Engine& engine, const I *demand,
    const std::vector<int>& misses, const State& state, int layer, const I *q) {
  const int n = int(misses.size());
  std::vector<int> columns, answer(n);
  for (int r = 0; r < W; ++r)
    for (I count = 0; count < q[r]; ++count) columns.push_back(r);
  if (int(columns.size()) != n) return {};
  if (engine.policy == 0) {
    std::vector<int> order(n); std::iota(order.begin(), order.end(), 0);
    for (int i = n - 1; i > 0; --i) {
      std::uint32_t mask = 1;
      while (mask < std::uint32_t(i)) mask = (mask << 1) | 1;
      std::uint32_t j;
      do { j = engine.random() & mask; } while (j > std::uint32_t(i));
      std::swap(order[i], order[j]);
    }
    for (int i = 0; i < n; ++i) answer[order[i]] = columns[i];
    return answer;
  }
  if (engine.policy == 7) {
    std::array<I,W> loads{}, remaining{};
    std::copy(q, q + W, remaining.begin());
    for (int e = 0; e < E; ++e) {
      const int bits = state.owner[layer * E + e];
      if (!bits) continue;
      I total = 0;
      for (int r = 0; r < W; ++r) total += demand[e * W + r];
      for (int r = 0; r < W; ++r) if (bits & (1 << r)) { loads[r] += total; break; }
    }
    std::vector<I> totals(n);
    for (int i = 0; i < n; ++i)
      for (int r = 0; r < W; ++r) totals[i] += demand[misses[i] * W + r];
    for (int i : descending_order(totals)) {
      std::array<I,W> projected; projected.fill(INF);
      I minimum = INF;
      for (int r = 0; r < W; ++r) if (remaining[r] > 0) {
        I maximum = 0;
        for (int other = 0; other < W; ++other)
          maximum = std::max(maximum, loads[other] + (other == r ? totals[i] : 0));
        projected[r] = maximum; minimum = std::min(minimum, maximum);
      }
      const I limit = minimum + std::max<I>(1, minimum * 200 / 10000);
      int best = -1; I best_local = -1, best_critical = INF, best_load = INF;
      for (int r = 0; r < W; ++r) {
        if (remaining[r] <= 0 || projected[r] > limit) continue;
        const I local = demand[misses[i] * W + r], dst = loads[r] + totals[i];
        if (local > best_local || (local == best_local && projected[r] < best_critical) ||
            (local == best_local && projected[r] == best_critical && dst < best_load) ||
            (local == best_local && projected[r] == best_critical && dst == best_load && (best < 0 || r < best))) {
          best = r; best_local = local; best_critical = projected[r]; best_load = dst;
        }
      }
      if (best < 0) return {};
      answer[i] = best; --remaining[best]; loads[best] += totals[i];
    }
    return answer;
  }
  // Preserve the legacy CA expanded-column Hungarian tie policy. For the
  // topology-weighted arm, replace local demand by negative measured cost.
  std::vector<I> scores(n * W), u(n+1), v(n+1), minimum(n+1);
  for (int i = 0; i < n; ++i) for (int dst = 0; dst < W; ++dst) {
    I value = demand[misses[i] * W + dst];
    if (engine.weighted) {
      value = 0;
      for (int origin = 0; origin < W; ++origin)
        value -= demand[misses[i] * W + origin] * engine.costs[origin * W + dst];
    }
    scores[i * W + dst] = value;
  }
  std::vector<int> p(n+1), way(n+1);
  std::vector<bool> used(n+1);
  for (int i = 1; i <= n; ++i) {
    p[0] = i; int j0 = 0;
    std::fill(minimum.begin(), minimum.end(), INF);
    std::fill(used.begin(), used.end(), false);
    do {
      used[j0] = true; const int i0 = p[j0]; I delta = INF; int j1 = 0;
      for (int j = 1; j <= n; ++j) if (!used[j]) {
        I value = -scores[(i0-1)*W+columns[j-1]] - u[i0] - v[j];
        if (value < minimum[j]) { minimum[j] = value; way[j] = j0; }
        if (minimum[j] < delta) { delta = minimum[j]; j1 = j; }
      }
      for (int j = 0; j <= n; ++j) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else minimum[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] != 0);
    do { const int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0 != 0);
  }
  for (int j = 1; j <= n; ++j) answer[p[j]-1] = columns[j-1];
  return answer;
}

extern "C" void* mgo_pcie_create(int seed, int policy, int quota_mode, const I* costs) noexcept {
  if ((policy != 0 && policy != 1 && policy != 7) || quota_mode < 0 || quota_mode > 1) return nullptr;
  try { return new Engine(seed, policy, quota_mode, costs); } catch (...) { return nullptr; }
}
extern "C" void mgo_pcie_destroy(void* handle) noexcept { delete static_cast<Engine*>(handle); }
extern "C" int mgo_pcie_trace(void* handle, double* output, int capacity) noexcept {
  if (!handle) return -1;
  const auto& trace = static_cast<Engine*>(handle)->trace;
  if (!output) return int(trace.size());
  if (capacity < int(trace.size())) return -1;
  for (std::size_t i = 0; i < trace.size(); ++i)
    std::copy(trace[i].begin(), trace[i].end(), output + i*61);
  return int(trace.size());
}
extern "C" int mgo_pcie_quotas(int n, int mode, int event, I* result) noexcept {
  if (n < 0 || event < 0 || !result || mode < 0 || mode > 1) return -1;
  quota(n, mode, event, result); return 0;
}

extern "C" int mgo_pcie_step(void *handle, int event, int n, int topk,
    const I *selected, const float *weights, const I *origins, const float *gate,
    State *s, Output *o) noexcept {
  try {
    if (!handle || event < 0 || n < 0 || topk < 1 || topk > 8 || s->stride < 1) return -1;
    auto& engine = *static_cast<Engine*>(handle);
    const int layer = event % L, tick = event + 1;
    double *row = o->row; std::fill(row, row + 48, 0);
    std::array<bool,E> active{}, resident{};
    std::array<float,E> maximum_weight{};
    std::array<I,E*W> demand{};
    std::copy(gate, gate + E, s->gates + layer * E);
    for (int e = 0; e < E; ++e) { o->targets[e] = e; resident[e] = s->owner[layer*E+e] != 0; }
    std::fill(o->effective, o->effective + n*topk, -1);
    std::fill(o->masses, o->masses + n*topk, 0);
    std::fill(o->lengths, o->lengths + n, 0);
    std::fill(o->destinations, o->destinations + n*topk, -1);
    for (int t = 0; t < n; ++t) {
      const int origin = int(origins[t]); if (origin < 0 || origin >= W) return -2;
      for (int k = 0; k < topk; ++k) {
        const int e = int(selected[t*topk+k]); if (e < 0 || e >= E) return -2;
        const float weight = weights[t*topk+k];
        if (!(weight >= 0) || weight > 1) return -2;
        active[e] = true; maximum_weight[e] = std::max(maximum_weight[e], weight);
        ++row[0]; row[1] += weight;
        const int bits = s->owner[layer*E+e];
        if (resident[e]) {
          ++row[2]; row[3] += weight; ++row[12]; row[13] += weight;
          if (bits & (1 << origin)) { ++row[4]; row[5] += weight; }
        } else { ++row[14]; row[15] += weight; }
        if (bits & (1 << origin)) { ++row[16]; row[17] += weight; }
        int position = -1;
        for (int j = 0; j < o->lengths[t]; ++j) if (o->effective[t*topk+j] == e) { position = j; break; }
        if (position < 0) {
          position = o->lengths[t]++; o->effective[t*topk+position] = e;
          ++demand[e*W+origin];
        }
        o->masses[t*topk+position] += weight;
      }
    }
    std::vector<int> misses;
    for (int e = 0; e < E; ++e) if (active[e]) {
      ++row[18];
      if (!resident[e]) { misses.push_back(e); ++row[19]; row[46] += maximum_weight[e] >= .20f; }
    }
    const int count = int(misses.size());
    quota(count, engine.quota_mode, event, o->quotas);
    auto assignment = assign(engine, demand.data(), misses, *s, layer, o->quotas);
    if (int(assignment.size()) != count) return -4;
    row[44] = *std::max_element(o->quotas, o->quotas + W);
    row[45] = *std::min_element(o->quotas, o->quotas + W);
    // Preflight all quota slots before modifying logical residency.
    for (int rank = 0; rank < W; ++rank) {
      int available = 0;
      for (int slot = 0; slot < s->capacities[rank]; ++slot) {
        const int key = s->slots[rank*s->stride+slot];
        available += key < 0 || key/E != layer || !active[key%E];
      }
      if (o->quotas[rank] > available) return -3;
    }
    for (int i = 0; i < count; ++i) {
      const int e = misses[i], rank = assignment[i], key = layer*E+e;
      int best = -1; float best_gate = std::numeric_limits<float>::infinity();
      int best_used = std::numeric_limits<int>::max(), best_key = std::numeric_limits<int>::max();
      for (int slot = 0; slot < s->capacities[rank]; ++slot) {
        const int candidate = s->slots[rank*s->stride+slot];
        if (candidate < 0) { best = slot; break; }
        if (candidate/E == layer && active[candidate%E]) continue;
        const float score = s->gates[candidate]; const int used = s->last[rank*K+candidate];
        if (score < best_gate || (score == best_gate && (used < best_used || (used == best_used && candidate < best_key)))) {
          best = slot; best_gate = score; best_used = used; best_key = candidate;
        }
      }
      if (best < 0) return -3;
      const int victim = s->slots[rank*s->stride+best];
      if (victim >= 0) { ++row[23]; s->owner[victim] = 0; s->primary[victim] = -1; s->lost[victim] = false; }
      if (s->owner[key]) return -5;
      if (s->seen[key]) { ++row[21]; row[33] += s->lost[key]; } else ++row[20];
      s->owner[key] = std::int16_t(1 << rank); s->primary[key] = std::int8_t(rank);
      s->slots[rank*s->stride+best] = key; s->last[rank*K+key] = tick;
      s->seen[key] = true; s->lost[key] = false;
      const I fetch[5] = {rank, key, best, victim, 0};
      std::copy(fetch, fetch+5, o->fetches+i*5);
    }
    std::array<bool,E*W> served{};
    std::array<double,61> record{};
    for (int t = 0; t < n; ++t) {
      const int origin = int(origins[t]); unsigned dispatch = 0;
      for (int j = 0; j < o->lengths[t]; ++j) {
        const int e = o->effective[t*topk+j], key = layer*E+e, dst = s->primary[key];
        if (dst < 0 || s->owner[key] != (1 << dst)) return -5;
        ++row[28]; row[32] += o->masses[t*topk+j]; served[dst*E+e] = true;
        ++record[57+dst];
        if (dst == origin) { ++row[27]; row[31] += o->masses[t*topk+j]; }
        else {
          ++row[29]; dispatch |= 1u << dst;
          ++record[origin/2 != dst/2 ? 53 : 54];
        }
        o->destinations[t*topk+j] = std::int8_t(dst);
      }
      row[30] += __builtin_popcount(dispatch);
      for (int dst=0; dst<W; ++dst) if (dispatch & (1u<<dst))
        ++record[origin/2 != dst/2 ? 55 : 56];
    }
    for (int rank = 0; rank < W; ++rank) {
      for (int e = 0; e < E; ++e) if (served[rank*E+e]) s->last[rank*K+layer*E+e] = tick;
      for (int slot = 0; slot < s->capacities[rank]; ++slot) row[24] += s->slots[rank*s->stride+slot] >= 0;
    }
    for (int key = 0; key < K; ++key) row[25] += s->owner[key] != 0;
    row[26] = row[24] - row[25];
    if (row[26] != 0 || row[27] + row[29] != row[28]) return -5;
    std::copy(row, row+48, record.begin());
    for (int rank=0; rank<W; ++rank) record[48+rank] = o->quotas[rank];
    record[52] = event;
    engine.trace.push_back(record);
    return count;
  } catch (...) { return -99; }
}
