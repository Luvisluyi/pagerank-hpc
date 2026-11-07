#include "graph.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>
#include <mpi.h>

CSRGraph generateSyntheticCSR(int N, int M, unsigned int seed) {
    CSRGraph graph;
    graph.num_nodes = N;
    graph.num_vertices = N;
    graph.num_edges = M;
    graph.out_degree.assign(N, 0);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, N - 1);

    std::vector<std::pair<int, int>> edges;
    edges.reserve(M);
    for (int i = 0; i < M; ++i) {
        int u = dist(rng);
        int v = dist(rng);
        edges.push_back({u, v});
        graph.out_degree[u]++;
    }

    std::sort(edges.begin(), edges.end());

    graph.row_ptr.assign(N + 1, 0);
    graph.col_idx.resize(M);

    for (const auto& edge : edges) {
        graph.row_ptr[edge.first + 1]++;
    }
    for (int i = 0; i < N; ++i) {
        graph.row_ptr[i + 1] += graph.row_ptr[i];
    }

    std::vector<int> current_pos = graph.row_ptr;
    for (const auto& edge : edges) {
        int u = edge.first;
        int v = edge.second;
        graph.col_idx[current_pos[u]++] = v;
    }

    return graph;
}

CSRGraph readEdgesToCSR(const std::string& filename) {
    CSRGraph graph;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return graph;
    }

    std::string line;
    int max_node = -1;
    std::vector<std::pair<int, int>> edges;

    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        int u, v;
        if (ss >> u >> v) {
            edges.push_back({u, v});
            max_node = std::max({max_node, u, v});
        }
    }
    infile.close();

    int N = max_node + 1;
    int M = static_cast<int>(edges.size());

    graph.num_nodes = N;
    graph.num_vertices = N;
    graph.num_edges = M;
    graph.out_degree.assign(N, 0);

    for (const auto& edge : edges) {
        graph.out_degree[edge.first]++;
    }

    std::sort(edges.begin(), edges.end());

    graph.row_ptr.assign(N + 1, 0);
    graph.col_idx.resize(M);

    for (const auto& edge : edges) {
        graph.row_ptr[edge.first + 1]++;
    }
    for (int i = 0; i < N; ++i) {
        graph.row_ptr[i + 1] += graph.row_ptr[i];
    }

    std::vector<int> current_pos = graph.row_ptr;
    for (const auto& edge : edges) {
        int u = edge.first;
        int v = edge.second;
        graph.col_idx[current_pos[u]++] = v;
    }

    return graph;
}

void broadCastGraph(CSRGraph& graph, int rank) {
    MPI_Bcast(&graph.num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&graph.num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&graph.num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        graph.row_ptr.resize(graph.num_nodes + 1);
        graph.col_idx.resize(graph.num_edges);
        graph.out_degree.resize(graph.num_nodes);
    }

    MPI_Bcast(graph.row_ptr.data(), graph.num_nodes + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(graph.col_idx.data(), graph.num_edges, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(graph.out_degree.data(), graph.num_nodes, MPI_INT, 0, MPI_COMM_WORLD);
}