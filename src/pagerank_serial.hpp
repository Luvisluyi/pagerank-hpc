#pragma once

#include "graph.hpp"
#include <vector>

void serialPageRank(const CSRGraph& graph, 
                    std::vector<double>& pr, 
                    double damping_factor = 0.85, 
                    double epsilon = 1e-5, 
                    int max_iter = 100);