#include "pagerank_hybrid.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>
#include <omp.h>


void openmp_PageRank(const CSRGraph& graph, std::vector<double>& global_pr, double damping_factor, double epsilon, int max_iter) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = graph.num_vertices;
    if (N == 0) return;

   
    int local_N = N / size + (rank < N % size ? 1 : 0);
    int local_start = rank * (N / size) + std::min(rank, N % size);
    int local_end = local_start + local_N;

    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = N / size + (i < N % size ? 1 : 0);
        displs[i] = i * (N / size) + std::min(i, N % size);
    }

    
    if (global_pr.size() != static_cast<size_t>(N)){ 
        global_pr.assign(N, 1.0 / N);
    }

    

    std::vector<double> local_pr_new(local_N, 0.0);

    int iter = 0;
    double global_diff = 1.0;
    double local_diff_prev = 0.0;
    
    double local_comp_time = 0.0;
    double local_comm_time = 0.0;
    double t_start, t_end;

    MPI_Barrier(MPI_COMM_WORLD);
    double total_start_time = MPI_Wtime();
    // Main PageRank Iteration
    
    while (iter < max_iter) {
        
        t_start=MPI_Wtime();
        double local_dangling = 0.0;

        #pragma omp parallel for reduction(+:local_dangling)
        for (int i = local_start; i < local_end; ++i) {
            if (graph.out_degree[i] == 0) {
                local_dangling += global_pr[i];
            }
        }
        t_end = MPI_Wtime();
        local_comp_time += (t_end-t_start);

        t_start = MPI_Wtime();
        double local_reductions[2]={local_dangling, local_diff_prev};
        double global_reductions[2]={0.0,0.0};
        MPI_Allreduce(local_reductions,global_reductions,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        t_end=MPI_Wtime();
        local_comm_time += (t_end - t_start);

        double global_dangling=global_reductions[0];

        if(iter>0){
            global_diff=global_reductions[1];
            if(global_diff<=epsilon) break;
        }

        t_start=MPI_Wtime();
        double base_pr =(1.0-damping_factor)/N + (damping_factor*global_dangling/N);
        double local_diff_current=0.0;



        #pragma omp parallel for schedule(dynamic,64) reduction(+:local_diff_current)
        for (int u = local_start; u<local_end; ++u){
            double sum = 0.0;
            int start = graph.row_ptr[u];
            int end = graph.row_ptr[u+1];

            for(int j = start; j<end; ++j){
                int v = graph.col_idx[j];
                if(graph.out_degree[v]>0){
                    sum+=global_pr[v]/ graph.out_degree[v];
                }
            }

            int local_idx = u - local_start;
            local_pr_new[local_idx]= base_pr + damping_factor *sum;
            local_diff_current += std::abs(local_pr_new[local_idx] - global_pr[u]);


        }

        local_diff_prev=local_diff_current;
        t_end = MPI_Wtime();
        local_comp_time += (t_end-t_start);

        t_start=MPI_Wtime();

        MPI_Allgatherv(local_pr_new.data(), local_N, MPI_DOUBLE,
                       global_pr.data(),recvcounts.data(),displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        t_end = MPI_Wtime();
        local_comm_time += (t_end-t_start);

        iter++;


    }

    double total_time= MPI_Wtime()-total_start_time;
    

    double max_total_time, max_comp_time, max_comm_time;
    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comm_time, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    
    double sum_comp_time, sum_comm_time;
    MPI_Reduce(&local_comp_time, &sum_comp_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comm_time, &sum_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double avg_comp_time = sum_comp_time / size;
        double avg_comm_time = sum_comm_time / size;
        
        
        double load_imbalance_ratio = max_comp_time / avg_comp_time;

        std::cout << "Hybrid MPI+OpenMP Converged in " << iter << " iterations." << std::endl;
        std::cout << "--- Profiling Results (MPI Ranks: " << size << " | Threads per Rank: " << omp_get_max_threads() << ") ---" << std::endl;
        std::cout << "Total Runtime (Max):  " << max_total_time << " s" << std::endl;
        
        std::cout << "\n[Computation]" << std::endl;
        std::cout << "Max Compute Time:     " << max_comp_time << " s" << std::endl;
        std::cout << "Avg Compute Time:     " << avg_comp_time << " s" << std::endl;
        std::cout << "Imbalance Ratio:      " << load_imbalance_ratio << "x" << std::endl;
        
        std::cout << "\n[Communication]" << std::endl;
        std::cout << "Max Comm Time:        " << max_comm_time << " s" << std::endl;
        std::cout << "Avg Comm Time:        " << avg_comm_time << " s" << std::endl;
        std::cout << "Network Wait Overhead:" << (max_comm_time - avg_comm_time) << " s" << std::endl;
    }
}