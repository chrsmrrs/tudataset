/**********************************************************************
 * Copyright (C) 2020 Christopher Morris <christopher.morris@udo.edu>
 *********************************************************************/


#include "ShortestPathKernel.h"

namespace ShortestPathKernel {
    ShortestPathKernel::ShortestPathKernel(const GraphDatabase &graph_database) : m_graph_database(graph_database),
                                                                                  m_distance_to_index(),
                                                                                  m_num_distances(0) {}

    GramMatrix ShortestPathKernel::compute_gram_matrix(bool use_labels, const bool compute_gram) {
        size_t num_graphs = m_graph_database.size();
        vector<DistanceCounter> distance_counters;

        // Compute shortest-path triple for each graph in graph database.
        for (const Graph &graph: m_graph_database) {
            DistanceTriples distances = compute_apsp(graph, use_labels);
            DistanceCounter distance_counter;

            for (const DistanceTriple d: distances) {
                double dis = get<0>(d);

                if ((dis > 0) and (dis != INT_MAX)) {
                    DistanceCounter::iterator it(distance_counter.find(d));
                    if (it == distance_counter.end()) {
                        distance_counter.insert({{d, 1}});
                        m_distance_to_index.insert({{d, m_num_distances}});
                        m_num_distances++;
                    } else {
                        it->second++;
                    }
                }
            }
            distance_counters.push_back((distance_counter));
        }

        // Compute feature vectors.
        vector<U> nonzero_compenents;
        for (Node i = 0; i < num_graphs; ++i) {
            DistanceCounter c = distance_counters[i];
            for (const auto &j: c) {
                DistanceTriple key = j.first;
                uint value = j.second;
                uint index = m_distance_to_index.find(key)->second;

                nonzero_compenents.push_back(U(i, index, value));
            }
        }

        // Compute gram matrix.
        GramMatrix feature_vectors(num_graphs, m_num_distances);
        feature_vectors.setFromTriplets(nonzero_compenents.begin(), nonzero_compenents.end());

        if (not compute_gram) {
            return feature_vectors;
        } else {
            GramMatrix gram_matrix(num_graphs, num_graphs);
            gram_matrix = feature_vectors * feature_vectors.transpose();

            return gram_matrix;
        }
    }

    DistanceTriples ShortestPathKernel::compute_apsp(const Graph &g, bool use_labels) {
        ShortestPathDistances shortest_path_distances;
        size_t num_nodes = g.get_num_nodes();

        vector<uint> distances;
        distances.resize(num_nodes, 0);
        shortest_path_distances.resize(num_nodes, distances);

        for (Node i = 0; i < num_nodes; ++i) {
            for (Node j = i; j < num_nodes; ++j) {
                if (g.has_edge(i, j)) {
                    shortest_path_distances[i][j] = 1;
                    shortest_path_distances[j][i] = 1;
                } else {
                    shortest_path_distances[i][j] = INT_MAX;
                    shortest_path_distances[j][i] = INT_MAX;
                }
            }
        }

        for (Node k = 0; k < num_nodes; ++k) {
            for (Node i = 0; i < num_nodes; ++i) {
                for (Node j = i; j < num_nodes; ++j) {
                    if ((shortest_path_distances[i][k] != INT_MAX) and (shortest_path_distances[k][j] != INT_MAX)) {
                        if (shortest_path_distances[i][j] >
                            shortest_path_distances[i][k] + shortest_path_distances[k][j]) {
                            shortest_path_distances[i][j] = shortest_path_distances[j][i] =
                                    shortest_path_distances[i][k] + shortest_path_distances[k][j];
                        }
                    }
                }
            }
        }

        Labels labels = g.get_labels();
        DistanceTriples triples;
        for (Node i = 0; i < num_nodes; ++i) {
            for (Node j = i; j < num_nodes; ++j) {
                if (use_labels) {
                    triples.push_back(make_tuple(labels[i], labels[j], shortest_path_distances[i][j]));
                    triples.push_back(make_tuple(labels[j], labels[i], shortest_path_distances[i][j]));
                } else {
                    triples.push_back(make_tuple(1, 1, shortest_path_distances[i][j]));
                    triples.push_back(make_tuple(1, 1, shortest_path_distances[i][j]));
                }

            }
        }

        return triples;
    }

    ShortestPathKernel::~ShortestPathKernel() {}
}

