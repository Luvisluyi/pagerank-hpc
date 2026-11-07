#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <vector>
#include <string>

struct CSRGraph {
    int num_nodes = 0;
    int num_vertices = 0;
    int num_edges = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<int> out_degree;
};

CSRGraph readEdgesToCSR(const std::string& filename);
inline CSRGraph loadGraphFromFile(const std::string& filename) { return readEdgesToCSR(filename); }
CSRGraph generateSyntheticCSR(int N, int M, unsigned int seed = 42);
void broadCastGraph(CSRGraph& graph, int rank);

#endif