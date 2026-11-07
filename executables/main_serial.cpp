#include <iostream>
#include <vector>
#include <string>

#include "graph.hpp"
#include "pagerank_serial.hpp"

int main(int argc, char** argv) {
    CSRGraph graph;

    std::cout << "=== Serial PageRank Execution ===" << std::endl;

    // Load file if provided as CLI argument, otherwise generate synthetic graph
    if (argc > 1) {
        std::string filename = argv[1];
        std::cout << "Loading graph dataset: " << filename << std::endl;
        graph = readEdgesToCSR(filename);
    } else {
        int N = 100000;
        int M = 1000000;
        std::cout << "No file provided. Generating synthetic graph (|V|=" << N << ", |E|=" << M << ")...\n";
        graph = generateSyntheticCSR(N, M);
    }

    std::vector<double> pr;
    std::cout << "Starting Serial PageRank computation...\n" << std::endl;
    
    serialPageRank(graph, pr);

    return 0;
}