// Capacity-constrained CA assignment: maximize local demand with exact
// per-rank quotas. Unlike expanded Hungarian columns, the graph has only
// rank nodes. This is a new tie policy; validate objective parity.
#include <cstdint>

extern "C" int mgo_ca_min_cost_flow(const std::int64_t* demand, int stride,
                                     const std::int64_t* experts, int n, int world,
                                     std::int64_t* answer) noexcept {
  if (!demand || !experts || !answer || n < 0 || n > 128 ||
      world < 1 || world > 8 || stride != world) return -1;
  if (n == 0) return 0;
  constexpr int max_nodes = 138;
  constexpr int max_edges = 2 * (128 + 128 * 8 + 8);
  constexpr std::int64_t inf = std::int64_t{1} << 50;
  const int sink = n + world + 1;
  int head[max_nodes], to[max_edges], next[max_edges], capacity[max_edges];
  std::int64_t cost[max_edges];
  for (int i = 0; i <= sink; ++i) head[i] = -1;
  int edges = 0;
  auto add = [&](int from, int dest, int cap, std::int64_t price) {
    to[edges] = dest; next[edges] = head[from]; head[from] = edges;
    capacity[edges] = cap; cost[edges++] = price;
    to[edges] = from; next[edges] = head[dest]; head[dest] = edges;
    capacity[edges] = 0; cost[edges++] = -price;
  };
  for (int i = 0; i < n; ++i) {
    const int expert = static_cast<int>(experts[i]);
    if (expert < 0 || expert >= 128) return -2;
    add(0, i + 1, 1, 0);
    for (int r = 0; r < world; ++r)
      add(i + 1, n + 1 + r, 1, -demand[expert * stride + r]);
  }
  for (int r = 0; r < world; ++r)
    add(n + 1 + r, sink, n / world + (r < n % world), 0);

  for (int sent = 0; sent < n; ++sent) {
    std::int64_t distance[max_nodes];
    int previous[max_nodes];
    bool queued[max_nodes] = {};
    int queue[max_nodes + 1], front = 0, back = 0;
    for (int v = 0; v <= sink; ++v) {
      distance[v] = inf;
      previous[v] = -1;
    }
    distance[0] = 0; queue[back++] = 0; queued[0] = true;
    while (front != back) {
      const int node = queue[front];
      front = (front + 1) % (max_nodes + 1);
      queued[node] = false;
      for (int edge = head[node]; edge != -1; edge = next[edge]) {
        if (!capacity[edge]) continue;
        const int dest = to[edge];
        const std::int64_t candidate = distance[node] + cost[edge];
        if (candidate < distance[dest]) {
          distance[dest] = candidate;
          previous[dest] = edge;
          if (!queued[dest]) {
            queue[back] = dest;
            back = (back + 1) % (max_nodes + 1);
            queued[dest] = true;
          }
        }
      }
    }
    if (previous[sink] < 0) return -3;
    for (int node = sink; node != 0;) {
      const int edge = previous[node];
      capacity[edge] -= 1;
      capacity[edge ^ 1] += 1;
      node = to[edge ^ 1];
    }
  }
  for (int i = 0; i < n; ++i) {
    answer[i] = -1;
    for (int edge = head[i + 1]; edge != -1; edge = next[edge]) {
      const int rank = to[edge] - (n + 1);
      if (rank >= 0 && rank < world && capacity[edge] == 0) {
        answer[i] = rank;
        break;
      }
    }
    if (answer[i] < 0) return -4;
  }
  return 0;
}
