#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "tsp_solver/tsp_solver.h"

void check(const std::vector<std::vector<int>>& costs, bool dummy,
           double expected_length, const char* name) {
  tsp_solver_ns::DataModel data;
  data.distance_matrix = costs;
  tsp_solver_ns::TSPSolver solver(data);
  solver.Solve();
  std::vector<int> route;
  solver.getSolutionNodeIndex(route, dummy);
  std::vector<int> sorted = route;
  std::sort(sorted.begin(), sorted.end());
  const int real_nodes = costs.size() - (dummy ? 1 : 0);
  if (sorted.size() != static_cast<size_t>(real_nodes)) throw std::runtime_error("route size");
  for (int i = 0; i < real_nodes; ++i) if (sorted[i] != i) throw std::runtime_error("route permutation");
  int cost = 0;
  for (size_t i = 1; i < route.size(); ++i) cost += costs[route[i-1]][route[i]];
  if (!dummy) cost += costs[route.back()][route.front()];
  if (std::abs(solver.getPathLength() - cost / 10.) > 1e-9 ||
      std::abs(solver.getPathLength() - expected_length) > 1e-9)
    throw std::runtime_error("reported and independently computed route costs differ");
  std::cout << "{\"fixture\":\"" << name << "\",\"path_length_m\":"
            << solver.getPathLength() << ",\"route\":[";
  for (size_t i=0; i<route.size(); ++i) std::cout << (i ? "," : "") << route[i];
  std::cout << "],\"dummy\":" << (dummy ? "true" : "false")
            << ",\"status\":\"passed\"}" << std::endl;
}

int main() {
  check({{0}}, false, 0., "single_node");
  check({{0,20},{20,0}}, false, 4., "two_nodes");
  check({{0,10,20,10},{10,0,10,20},{20,10,0,10},{10,20,10,0}},
        false, 4., "square_closed_tour");
  check({{0,10,20,30,0},{10,0,10,20,0},{20,10,0,10,0},
         {30,20,10,0,0},{0,0,0,0,0}}, true, 3., "line_open_tour_dummy");
}
