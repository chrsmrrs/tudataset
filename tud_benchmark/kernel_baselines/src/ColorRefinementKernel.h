/**********************************************************************
 * Copyright (C) 2020 Christopher Morris <christopher.morris@udo.edu>
 *********************************************************************/


#ifndef WLFAST_COLORREFINEMENTKERNEL_H
#define WLFAST_COLORREFINEMENTKERNEL_H

#include <algorithm>
#include <unordered_map>

#include <iostream>
#include <fstream>
#include <string>

#include "AuxiliaryMethods.h"
#include "Graph.h"


using namespace GraphLibrary;

namespace ColorRefinement {
    class ColorRefinementKernel {
    public:
        ColorRefinementKernel(const GraphDatabase &graph_database);

        // Computes gram matrix for the Weisfeiler-Lehman subtree kernel.
        GramMatrix compute_gram_matrix(const uint num_iterations, const bool use_labels, const bool use_edge_labels, const bool compute_gram, const bool wloa);

        ~ColorRefinementKernel();

    private:
        // Computes labels for vertices of graph.
        ColorCounter compute_colors(const Graph &g, const uint num_iterations, bool use_labels, bool use_edge_labels);

        // Manages graphs.
        GraphDatabase m_graph_database;

        // Manage indices of of labels in feature vectors.
        ColorCounter m_label_to_index;

        // Counts number of distinct labels over all graphs.
        uint m_num_labels;
    };
}

#endif //WLFAST_COLORREFINEMENTKERNEL_H
