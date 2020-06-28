/**********************************************************************
 * Copyright (C) 2020 Christopher Morris <christopher.morris@udo.edu>
 *********************************************************************/


#ifndef WLFAST_GRAPH_H
#define WLFAST_GRAPH_H

#ifdef __linux__
#include <eigen3/Eigen/Sparse>
#else
#include "/usr/local/include/eigen3/Eigen/Sparse"
#endif




#include <unordered_set>
#include <unordered_map>
#include <vector>

using Eigen::SparseMatrix;
using namespace std;

using uint = unsigned int;
using Node = uint;
using Nodes = vector<Node>;
using Label = unsigned long;
using Labels = vector<Label>;
using Edge = tuple<Node, Node>;
using EdgeLabels = unordered_map<Edge, uint>;
using EdgeList = vector<Edge>;
using SpMatrix = Eigen::SparseMatrix<double>;
using GramMatrix = SpMatrix;
using AdjacenyMatrix = SpMatrix;
using ColorCounter = map<Label, uint>;
using S = Eigen::Triplet<double>;

using TwoTuple = tuple<Node, Node>;
using ThreeTuple = tuple<Node, Node, Node>;

namespace std {
    namespace {
        // Code from boost: Reciprocal of the golden ratio helps spread entropy and handles duplicates.
        // See Mike Seymour in magic-numbers-in-boosthash-combine: http://stackoverflow.com/questions/4948780 .
        template<class T>
        inline void hash_combine(std::size_t &seed, T const &v) {
            seed ^= hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }

        // Recursive template code derived from Matthieu M.
        template<class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
        struct HashValueImpl {
            static void apply(size_t &seed, Tuple const &tuple) {
                HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
                hash_combine(seed, get<Index>(tuple));
            }
        };

        template<class Tuple>
        struct HashValueImpl<Tuple, 0> {
            static void apply(size_t &seed, Tuple const &tuple) {
                hash_combine(seed, get<0>(tuple));
            }
        };
    }

    template<typename ... TT>
    struct hash<std::tuple<TT...>> {
        size_t
        operator()(std::tuple<TT...> const &tt) const {
            size_t seed = 0;
            HashValueImpl<std::tuple<TT...> >::apply(seed, tt);
            return seed;
        }
    };
}

namespace GraphLibrary {
    class Graph {
    public:
        Graph(const bool directed);

        Graph(const bool directed, const uint num_nodes, const EdgeList &edgeList, const Labels node_labels);

        // Add a single node to the graph.
        size_t add_node();

        // Add a single edge to the graph.
        void add_edge(const Node v, const Node w);

        // Get degree of node "v".
        size_t get_degree(const Node v) const;

        // Get neighbors of node "v".
        Nodes get_neighbours(const Node v) const;

        // Get number of nodes in graph.
        size_t get_num_nodes() const;

        // Get number of edges in graph.
        size_t get_num_edges() const;

        // Returns "1" if edge {u,w} exists, otherwise "0".
        uint has_edge(const Node v, const Node w) const;

        // Get node labels of graphs.
        Labels get_labels() const;

        // Set node labels of graphs.
        void set_labels(Labels &labels);

        // Get edge labels of graphs.
        EdgeLabels get_edge_labels() const;

        EdgeLabels get_vertex_id() const;

        // Set edge labels of graphs.
        void set_edge_labels(EdgeLabels &labels);

        void set_vertex_id(EdgeLabels &vertex_id);

        void set_local(EdgeLabels &local);

        void set_node_to_two_tuple(unordered_map<Node, TwoTuple> &n);
        unordered_map<Node, TwoTuple> get_node_to_two_tuple() const;

        void set_node_to_three_tuple(unordered_map<Node, ThreeTuple> &n);
        unordered_map<Node, ThreeTuple> get_node_to_three_tuple() const;

        EdgeLabels get_local() const;

        // Manage node labels.
        Labels m_node_labels;
        EdgeLabels m_edge_labels;
        EdgeLabels m_vertex_id;
        EdgeLabels m_local;
        unordered_map<Node, TwoTuple> m_node_to_two_tuple;
        unordered_map<Node, ThreeTuple> m_node_to_three_tuple;

        ~Graph();

    private:
        vector<vector<Node>> m_adjacency_lists;

        // Manage number of nodes in graph.
        size_t m_num_nodes;
        // Manage number of edges in graph.
        size_t m_num_edges;
        // true if graph is directed.
        bool m_is_directed;
    };

    typedef vector<Graph> GraphDatabase;
}

#endif //WLFAST_GRAPH_H
