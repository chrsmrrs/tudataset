/**********************************************************************
 * Copyright (C) 2017 Christopher Morris <christopher.morris@udo.edu>
 *
 * This file is part of globalwl.
 *
 * globalwl can not be copied and/or distributed without the express
 * permission of Christopher Morris.
 *********************************************************************/

#ifndef WLFAST_SHORTESTPATHKERNEL_H
#define WLFAST_SHORTESTPATHKERNEL_H


#include <unordered_map>

#include "Graph.h"

using ShortestPathDistances = vector<vector<uint>>;
using DistanceTriple = tuple<uint, Label, Label>;
using DistanceTriples = vector<DistanceTriple>;
using DistanceCounter = unordered_map<DistanceTriple, uint>;
using U = Eigen::Triplet<uint>;

using namespace GraphLibrary;

namespace ShortestPathKernel {
    class ShortestPathKernel {
    public:
        ShortestPathKernel(const GraphDatabase &graph_database);

        // Computes gram matrix for the Weisfeiler-Lehman subtree kernel.
        GramMatrix compute_gram_matrix(bool use_labels, const bool compute_gram);

        ~ShortestPathKernel();

    private:
        // Computes shortest-path triples for each graph using Floyd-Warshall algorithm.
        DistanceTriples compute_apsp(const Graph &g, bool use_labels);

        // Manages graphs.
        GraphDatabase m_graph_database;

        // Manage indices of of distance triples in feature vectors.
        DistanceCounter m_distance_to_index;

        // Counts number of distinct distance triples over all graphs.
        uint m_num_distances;
    };
}

#endif //WLFAST_SHORTESTPATHKERNEL_H
