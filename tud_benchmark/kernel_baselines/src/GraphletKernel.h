/**********************************************************************
 * Copyright (C) 2020 Christopher Morris <christopher.morris@udo.edu>
 *********************************************************************/


#ifndef WLFAST_GRAPHLETKERNEL_H
#define WLFAST_GRAPHLETKERNEL_H

#include <algorithm>
#include <unordered_map>

#include "AuxiliaryMethods.h"
#include "Graph.h"

using Graphlet = Label;
using GraphletCounter = map<Graphlet, uint>;

using namespace GraphLibrary;

namespace GraphletKernel {
    class GraphletKernel {
    public:
        GraphletKernel(const GraphDatabase &graph_database);

        // Computes gram matrix for the graphlet kernel.
        GramMatrix compute_gram_matrix(const bool use_labels, const bool use_edge_labels, const bool compute_gram);

        ~GraphletKernel();

    private:
        // Computes number of graphlets in graph.
        GraphletCounter compute_graphlet_count(const Graph &g, const bool use_labels, const bool use_edge_labels);

        // Manages graphs.
        GraphDatabase m_graph_database;

        // Manage indices of labels in feature vectors.
        ColorCounter m_label_to_index;

        // Counts number of distinct labels over all graphs.
        uint m_num_labels;
    };
}
#endif //WLFAST_GRAPHLETKERNEL_H
