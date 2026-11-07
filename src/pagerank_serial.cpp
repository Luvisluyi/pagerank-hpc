#include "pagerank_serial.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>
#include <omp.h>


//Serial Execution

void serialPageRank(const CSRGraph& graph, std::vector<double>& pr, double damping_factor, double epsilon, int max_iter) {
    int N = graph.num_vertices;
    if (N == 0) return;

    pr.assign(N, 1.0 / N);
    std::vector<double> pr_new(N, 0.0);

    int iter = 0;
    double diff = 1.0;

    auto start_time=std::chrono::high_resolution_clock::now();

    while (iter < max_iter && diff > epsilon) {
        diff = 0.0;
        
        // dangling nodes
        double dangling_sum = 0.0;
        for (int i = 0; i < N; ++i) {
            if (graph.out_degree[i] == 0) {
                dangling_sum += pr[i];
            }
        }

      
        double base_pr = (1.0 - damping_factor) / N + (damping_factor * dangling_sum / N);

        
        for (int u = 0; u < N; ++u) {
            double sum = 0.0;
            int start = graph.row_ptr[u];
            int end = graph.row_ptr[u + 1];

            for (int j = start; j < end; ++j) {
                int v = graph.col_idx[j]; 
                if (graph.out_degree[v] > 0) {
                    sum += pr[v] / graph.out_degree[v];
                }
            }

            pr_new[u] = base_pr + damping_factor * sum;
            diff += std::abs(pr_new[u] - pr[u]);
        }

        
        std::swap(pr, pr_new);
        iter++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Serial Converged in " << iter << " iterations." << std::endl;
    std::cout << ":Serial Runtime: " << elapsed.count() << " seconds." << std::endl;
}


