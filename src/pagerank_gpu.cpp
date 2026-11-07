#include "pagerank_gpu.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <mpi.h>
#include <omp.h>


void openmp_gpu_PageRank(const CSRGraph& graph, std::vector<double>& global_pr, double damping_factor, double epsilon, int max_iter) {
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


    const int* row_ptr_d = graph.row_ptr.data();
    const int* col_idx_d = graph.col_idx.data();
    const int* out_degree_d = graph.out_degree.data();
    double* global_pr_d = global_pr.data();
    double* local_pr_new_d = local_pr_new.data();
    int num_edges = graph.num_edges;
    
    
    double local_gpu_compute = 0.0;
    double local_pcie_transfer = 0.0;
    double local_mpi_comm = 0.0;
    double t_start, t_end;

    MPI_Barrier(MPI_COMM_WORLD);
    double total_start_time = MPI_Wtime();
    
    // Main PageRank Iteration

    #pragma omp target data map(to: row_ptr_d[0: N+1],\
                                    col_idx_d[0 : num_edges], \
                                    out_degree_d[0 : N]) \
                            map(alloc: global_pr_d[0 : N], \
                                       local_pr_new_d[0 : local_N])
    {
        while (iter < max_iter) {

            t_start=MPI_Wtime();
            #pragma omp target update to(global_pr_d[0 : N])
            t_end=MPI_Wtime();
            local_pcie_transfer += (t_end-t_start);
        
            double local_dangling = 0.0;

            t_start=MPI_Wtime();
            #pragma omp target teams distribute parallel for reduction(+:local_dangling)
            for (int i = local_start; i < local_end; ++i) {
                if (out_degree_d[i] == 0) {
                    local_dangling += global_pr_d[i];
                }
            }
            t_end=MPI_Wtime();
            local_gpu_compute += (t_end- t_start);

            t_start=MPI_Wtime();
            double local_reductions[2]={local_dangling, local_diff_prev};
            double global_reductions[2]={0.0,0.0};
            MPI_Allreduce(local_reductions,global_reductions,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            t_end = MPI_Wtime();
            local_mpi_comm += (t_end - t_start);

            double global_dangling=global_reductions[0];

            if(iter>0){
                global_diff=global_reductions[1];
                if(global_diff<=epsilon) break;
            }

            double base_pr =(1.0-damping_factor)/N + (damping_factor*global_dangling/N);

            double local_diff_current=0.0;


            t_start=MPI_Wtime();
            #pragma omp target teams distribute parallel for reduction(+:local_diff_current)
            for (int u = local_start; u<local_end; ++u){
                double sum = 0.0;
                int start = row_ptr_d[u];
                int end = row_ptr_d[u+1];

                for(int j = start; j<end; ++j){
                    int v = col_idx_d[j];
                    if(out_degree_d[v]>0){
                        sum+=global_pr_d[v]/ out_degree_d[v];
                    }
                }

                int local_idx = u - local_start;
                local_pr_new_d[local_idx]= base_pr + damping_factor *sum;
                local_diff_current += std::abs(local_pr_new_d[local_idx] - global_pr_d[u]);


            }

            local_diff_prev=local_diff_current;
            t_end=MPI_Wtime();
            local_gpu_compute += (t_end-t_start);
            
            t_start=MPI_Wtime();
            #pragma omp target update from(local_pr_new_d[0: local_N])
            t_end=MPI_Wtime();
            local_pcie_transfer +=  (t_end-t_start);

            t_start=MPI_Wtime();
            MPI_Allgatherv(local_pr_new_d,local_N,MPI_DOUBLE,
                           global_pr_d,recvcounts.data(),displs.data(),
                           MPI_DOUBLE,MPI_COMM_WORLD);
            t_end = MPI_Wtime();
            local_mpi_comm += (t_end-t_start);
            
            
            iter++;


        }
    }
    
    double total_time = MPI_Wtime() - total_start_time;
    
    
    double max_total, max_gpu, max_pcie, max_mpi;
    MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_gpu_compute, &max_gpu, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_pcie_transfer, &max_pcie, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_mpi_comm, &max_mpi, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    
    double sum_gpu;
    MPI_Reduce(&local_gpu_compute, &sum_gpu, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double avg_gpu = sum_gpu / size;
        double load_imbalance = max_gpu / avg_gpu;

        std::cout << "Hybrid MPI+GPU Converged in " << iter << " iterations." << std::endl;
        std::cout << "--- Profiling Results (MPI Ranks: " << size << " | GPUs per Node Enabled) ---" << std::endl;
        std::cout << "Total Runtime (Max):  " << max_total << " s\n" << std::endl;
        
        std::cout << "[GPU Compute]" << std::endl;
        std::cout << "Max Compute Time:     " << max_gpu << " s" << std::endl;
        std::cout << "Avg Compute Time:     " << avg_gpu << " s" << std::endl;
        std::cout << "GPU Imbalance Ratio:  " << load_imbalance << "x" << std::endl;
        
        std::cout << "\n[Overheads]" << std::endl;
        std::cout << "Max PCIe Transfer:    " << max_pcie << " s" << std::endl;
        std::cout << "Max MPI Network Comm: " << max_mpi << " s" << std::endl;
    }
}