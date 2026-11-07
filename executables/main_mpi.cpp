#include <iostream>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include "graph.hpp"
#include "pagerank_mpi.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Default graph parameters for strong scaling if CLI args are omitted
    int num_nodes = 10000000;
    int num_edges = 100000000;

    if (argc >= 3) {
        num_nodes = std::atoi(argv[1]);
        num_edges = std::atoi(argv[2]);
    }

    CSRGraph graph;
    if (rank == 0) {
        std::cout << "=== Pure MPI PageRank Execution ===" << std::endl;
        std::cout << "Running on " << num_procs << " MPI ranks." << std::endl;
        std::cout << "Graph Parameters: |V| = " << num_nodes 
                  << ", |E| = " << num_edges << std::endl;
        graph = generateSyntheticCSR(num_nodes, num_edges);
    }

    // Broadcast graph structure from Rank 0 to worker ranks
    broadCastGraph(graph, rank);

    // Execute MPI PageRank
    std::vector<double> global_pr;
    mpiOnlyPageRank(graph, global_pr, 0.85, 1e-6);

    MPI_Finalize();
    return 0;
}