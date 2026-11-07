#include <iostream>
#include <vector>
#include <string>
#include <mpi.h>
#include <omp.h>

#include "graph.hpp"
#include "pagerank_hybrid.hpp"

int main(int argc, char** argv) {
    int provided;
    // Require FUNNELED thread support for Hybrid MPI + OpenMP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED && rank == 0) {
        std::cerr << "Warning: MPI implementation does not provide MPI_THREAD_FUNNELED level." << std::endl;
    }

    
    CSRGraph graph;

    if (rank == 0) {
        std::cout << "=== Hybrid MPI + OpenMP (CPU) PageRank Execution ===" << std::endl;
        std::cout << "Running on " << size << " MPI ranks with " 
                  << omp_get_max_threads() << " OpenMP threads per rank." << std::endl;

        if (argc > 1) {
            std::string filename = argv[1];
            std::cout << "Loading graph dataset: " << filename << std::endl;
            graph = loadGraphFromFile(filename);
        } else {
            int N = 100000;
            int M = 1000000;
            std::cout << "No file provided. Generating synthetic graph (|V|=" << N << ", |E|=" << M << ")...\n";
            graph = generateSyntheticCSR(N, M);
        }
    }

    broadCastGraph(graph, rank);

    std::vector<double> global_pr;

    if (rank == 0) std::cout << "Starting Hybrid CPU PageRank computation...\n" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    openmp_PageRank(graph, global_pr);

    MPI_Finalize();
    return 0;
}