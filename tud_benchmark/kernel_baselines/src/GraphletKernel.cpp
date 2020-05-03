/**********************************************************************
 * Copyright (C) 2017 Christopher Morris <christopher.morris@udo.edu>
 *
 * This file is part of globalwl.
 *
 * globalwl can not be copied and/or distributed without the express
 * permission of Christopher Morris.
 *********************************************************************/

#include "GraphletKernel.h"

namespace GraphletKernel {
    GraphletKernel::GraphletKernel(const GraphDatabase &graph_database) : m_graph_database(graph_database),
                                                                          m_label_to_index(),
                                                                          m_num_labels(0) {}

    GramMatrix
    GraphletKernel::compute_gram_matrix(const bool use_labels, const bool use_edge_labels, const bool compute_gram) {
        vector <GraphletCounter> graphlet_counters;
        graphlet_counters.reserve(m_graph_database.size());

        // Compute graphlet count for each graph in graph database.
        for (Graph &graph: m_graph_database) {
            graphlet_counters.push_back(compute_graphlet_count(graph, use_labels, use_edge_labels));
        }

        size_t num_graphs = m_graph_database.size();
        vector <S> nonzero_compenents;

        // Compute feature vector.
        for (Node i = 0; i < num_graphs; ++i) {
            GraphletCounter c = graphlet_counters[i];
            for (const auto &j: c) {
                Label key = j.first;
                // Divide by six to not double count graphlets.
                double value = j.second / 6.0;
                uint index = m_label_to_index.find(key)->second;
                nonzero_compenents.push_back(S(i, index, value));
            }
        }

        // Compute gram matrix.
        GramMatrix feature_vectors(num_graphs, m_num_labels);
        feature_vectors.setFromTriplets(nonzero_compenents.begin(), nonzero_compenents.end());

        if (not compute_gram) {
            return feature_vectors;
        } else {
            GramMatrix gram_matrix(num_graphs, num_graphs);
            gram_matrix = feature_vectors * feature_vectors.transpose();

            return gram_matrix;
        }
    }

    GraphletCounter
    GraphletKernel::compute_graphlet_count(const Graph &g, const bool use_labels, const bool use_edge_labels) {
        GraphletCounter graphlet_counter;

        size_t num_nodes = g.get_num_nodes();
        Labels labels;
        labels.reserve(num_nodes);

        if (use_labels) {
            labels = g.get_labels();
        }

        EdgeLabels edge_labels;
        if (use_edge_labels) {
            edge_labels = g.get_edge_labels();
        }

        // Generate all connected 3-node graplets.
        // We search for every path of length three, and then distinguish if it is a triangle or a wedge.
        for (Node u = 0; u < num_nodes; ++u) {
            Nodes u_neighbors = g.get_neighbours(u);
            for (const Node v: u_neighbors) {
                Nodes v_neighbors = g.get_neighbours(v);
                for (const Node w: v_neighbors) {
                    Label new_label;
                    if (w != u) {
                        // Found triangle.
                        if (g.has_edge(u, w)) {
                            if (use_labels) {
                                new_label = 3;
                                Label l_u = labels[u];
                                Label l_v = labels[v];
                                Label l_w = labels[w];

                                Labels new_labels;

                                if (use_edge_labels) {
                                    uint uv = edge_labels.find(make_tuple(u, v))->second;
                                    uint uw = edge_labels.find(make_tuple(u, w))->second;
                                    uint vw = edge_labels.find(make_tuple(v, w))->second;

                                    new_labels(
                                            // All vertices have degree two.
                                            {{AuxiliaryMethods::pairing(2, l_u),
                                                     AuxiliaryMethods::pairing(uv, l_u),
                                                     AuxiliaryMethods::pairing(uw, l_u),
                                                     AuxiliaryMethods::pairing(2, l_v),
                                                     AuxiliaryMethods::pairing(2, l_w),
                                                     AuxiliaryMethods::pairing(vw, l_v)
                                             }});


                                } else {
                                    // Map every labeled triangle to a unique integer.
                                    new_labels(
                                            {{AuxiliaryMethods::pairing(2,
                                                                        l_u), AuxiliaryMethods::pairing(
                                                    2,
                                                    l_v), AuxiliaryMethods::pairing(
                                                    2,
                                                    l_w)}});
                                }

                                sort(new_labels.begin(), new_labels.end());
                                for (Label d: new_labels) {
                                    new_label = AuxiliaryMethods::pairing(new_label, d);
                                }

                                // No labels.
                            } else {
                                new_label = 3;
                            }

                            GraphletCounter::iterator it(graphlet_counter.find(new_label));
                            if (it == graphlet_counter.end()) {
                                graphlet_counter.insert({{new_label, 1}});
                                m_label_to_index.insert({{new_label, m_num_labels}});
                                m_num_labels++;
                            } else {
                                it->second++;
                            }
                        } else { // Found wedge.
                            if (use_labels) {
                                new_label = 2;
                                Label l_u = labels[u];
                                Label l_v = labels[v];
                                Label l_w = labels[w];

                                Labels new_labels;

                                if (use_edge_labels) {
                                    uint uv = edge_labels.find(make_tuple(u, v))->second;
                                    uint vw = edge_labels.find(make_tuple(v, w))->second;

                                    new_labels(
                                            {{AuxiliaryMethods::pairing(1, l_u),
                                                     AuxiliaryMethods::pairing(uv, l_u),
                                                     AuxiliaryMethods::pairing(2, l_v),
                                                     AuxiliaryMethods::pairing(1, l_w),
                                                     AuxiliaryMethods::pairing(vw, l_v)
                                             }});
                                } else {

                                    // Map every labeled triangle to a unique integer.
                                    new_labels(
                                            {{AuxiliaryMethods::pairing(1,
                                                                        l_u), AuxiliaryMethods::pairing(
                                                    2, l_v), AuxiliaryMethods::pairing(
                                                    1, l_w)}});
                                }

                                sort(new_labels.begin(), new_labels.end());
                                for (Label d: labels) {
                                    new_label = AuxiliaryMethods::pairing(new_label, d);
                                }


                            } else {
                                new_label = 2;
                            }

                            GraphletCounter::iterator it(graphlet_counter.find(new_label));
                            if (it == graphlet_counter.end()) {
                                graphlet_counter.insert({{new_label, 3}});
                                m_label_to_index.insert({{new_label, m_num_labels}});
                                m_num_labels++;
                            } else {
                                it->second = it->second + 3;
                            }
                        }
                    }
                }
            }
        }

        return graphlet_counter;
    }

    GraphletKernel::~GraphletKernel() {}
}