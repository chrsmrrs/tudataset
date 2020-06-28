/**********************************************************************
 * Copyright (C) 2020 Christopher Morris <christopher.morris@udo.edu>
 *********************************************************************/

#include "Graph.h"

namespace GraphLibrary {
    Graph::Graph(const bool directed) : m_is_directed(directed), m_num_nodes(0), m_num_edges(0), m_node_labels(),
                                        m_edge_labels(), m_vertex_id(), m_local(), m_node_to_two_tuple(), m_node_to_three_tuple() {}

    Graph::Graph(const bool directed, const uint num_nodes, const EdgeList &edgeList, const Labels node_labels)
            : m_is_directed(directed), m_adjacency_lists(), m_num_nodes(num_nodes), m_num_edges(edgeList.size()),
              m_node_labels(node_labels), m_edge_labels(), m_vertex_id(), m_local(), m_node_to_two_tuple(), m_node_to_three_tuple() {
        m_adjacency_lists.resize(num_nodes);

        for (auto const &e: edgeList) {
            add_edge(get<0>(e), get<1>(e));
        }
    }

    size_t Graph::add_node() {
        vector<Node> new_node;
        m_adjacency_lists.push_back(move(new_node));
        m_num_nodes++;

        return m_num_nodes - 1;
    }

    void Graph::add_edge(const Node v, const Node w) {

        if (!m_is_directed) {
            m_adjacency_lists[v].push_back(w);
            m_adjacency_lists[w].push_back(v);
        } else {
            m_adjacency_lists[v].push_back(w);
        }


        if (!m_is_directed) {
            m_num_edges++;
        } else {
            m_num_edges += 2;
        }
    }

    size_t Graph::get_degree(const Node v) const {
        return m_adjacency_lists[v].size();
    }

    Nodes Graph::get_neighbours(const Node v) const {
        return m_adjacency_lists[v];
    }

    size_t Graph::get_num_nodes() const {
        return m_num_nodes;
    }

    size_t Graph::get_num_edges() const {
        return m_num_edges;
    }

    uint Graph::has_edge(const Node v, const Node w) const {
        // This works for directed as well as undirected graphs.
        if (find(m_adjacency_lists[v].begin(), m_adjacency_lists[v].end(), w) != m_adjacency_lists[v].end()) {
            return 1;
        } else {
            return 0;
        }

    }

    Labels Graph::get_labels() const {
        return m_node_labels;
    }

    void Graph::set_labels(Labels &labels) {
        // Copy labels.
        m_node_labels = labels;
    }


    void Graph::set_edge_labels(EdgeLabels &labels) {
        // Copy labels.
        m_edge_labels = labels;
    }


    EdgeLabels Graph::get_edge_labels() const {
        return m_edge_labels;
    }

    EdgeLabels Graph::get_vertex_id() const {
        return m_vertex_id;
    }


    void Graph::set_vertex_id(EdgeLabels &vertex_id) {
        // Copy labels.
        m_vertex_id = vertex_id;
    }

    EdgeLabels Graph::get_local() const {
        return m_local;
    }


    void Graph::set_local(EdgeLabels &local) {
        // Copy labels.
        m_local = local;
    }

    void Graph::set_node_to_two_tuple(unordered_map<Node, TwoTuple> &n) {
        m_node_to_two_tuple = n;
    }

    void Graph::set_node_to_three_tuple(unordered_map<Node, ThreeTuple> &n) {
        m_node_to_three_tuple = n;
    }

    unordered_map<Node, TwoTuple> Graph::get_node_to_two_tuple() const {
        return m_node_to_two_tuple;
    }

    unordered_map<Node, ThreeTuple> Graph::get_node_to_three_tuple() const {
        return m_node_to_three_tuple;
    }

    Graph::~Graph() {}
}