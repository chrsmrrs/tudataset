import os.path as path
import sys

import networkx as nx

sys.path.insert(0, '..')
sys.path.insert(0, '.')


def read_txt(ds_name):
    pre = ""

    with open("./datasets/" + pre + ds_name + "/" + ds_name + "/" + "raw/" + ds_name + "_graph_indicator.txt",
              "r") as f:
        graph_indicator = [int(i) - 1 for i in list(f)]
    f.closed

    # Nodes.
    num_graphs = max(graph_indicator)
    node_indices = []
    offset = []
    c = 0

    for i in range(num_graphs + 1):
        offset.append(c)
        c_i = graph_indicator.count(i)
        node_indices.append((c, c + c_i - 1))
        c += c_i

    graph_db = []
    for i in node_indices:
        g = nx.Graph()
        for j in range(i[1] - i[0]):
            g.add_node(j + 1)

        graph_db.append(g)

    # Edges.
    with open("./datasets/" + pre + ds_name + "/" + ds_name + "/" + "raw/" + ds_name + "_A.txt", "r") as f:
        edges = [i.split(',') for i in list(f)]
    f.closed

    edges = [(int(e[0].strip()) - 1, int(e[1].strip()) - 1) for e in edges]
    edge_list = []
    for e in edges:
        g_id = graph_indicator[e[0]]
        g = graph_db[g_id]
        off = offset[g_id]

        # Avoid multigraph (for edge_list)
        if ((e[0] - off, e[1] - off) not in list(g.edges())) and ((e[1] - off, e[0] - off) not in list(g.edges())):
            g.add_edge(e[0] - off, e[1] - off)
            edge_list.append((e[0] - off, e[1] - off))

    # Node labels.
    if path.exists("./datasets/" + pre + ds_name + "/" + ds_name + "/" + "raw/" + ds_name + "_node_labels.txt"):
        with open("./datasets/" + pre + ds_name + "/" + ds_name + "/" + "raw/" + ds_name + "_node_labels.txt",
                  "r") as f:
            node_labels = [map(int, i.split(',')) for i in list(f)]  # [int(i) for i in list(f)]
        f.closed

        i = 0
        for g in graph_db:
            for v in range(g.number_of_nodes()):
                g.nodes[v]['labels'] = node_labels[i]
                i += 1

    # Node Attributes.
    if path.exists("./datasets/" + pre + ds_name + "/" + ds_name + "/" + "raw/" + ds_name + "_node_attributes.txt"):
        with open("./datasets/" + pre + ds_name + "/" + ds_name + "/" + "raw/" + ds_name + "_node_attributes.txt",
                  "r") as f:
            node_attributes = [map(float, i.split(',')) for i in list(f)]
        f.closed

        i = 0
        for g in graph_db:
            for v in range(g.number_of_nodes()):
                g.nodes[v]['attributes'] = node_attributes[i]
                i += 1

    # Edge Labels.
    if path.exists("./datasets/" + pre + ds_name + "/" + ds_name + "/" + "raw/" + ds_name + "_edge_labels.txt"):
        with open("./datasets/" + pre + ds_name + "/" + ds_name + "/" + "raw/" + ds_name + "_edge_labels.txt",
                  "r") as f:
            edge_labels = [map(int, i.split(',')) for i in list(f)]  # [int(i) for i in list(f)]
        f.closed

        i = 0
        for g in graph_db:
            for e in range(g.number_of_edges()):
                g.edges[edge_list[i]]['labels'] = edge_labels[i]
                i += 1

    # Edge Attributes.
    if path.exists("./datasets/" + pre + ds_name + "/" + ds_name + "/" + "raw/" + ds_name + "_edge_attributes.txt"):
        with open("./datasets/" + pre + ds_name + "/" + ds_name + "/" + "raw/" + ds_name + "_edge_attributes.txt",
                  "r") as f:
            edge_attributes = [map(float, i.split(',')) for i in list(f)]
        f.closed

        i = 0
        for g in graph_db:
            for e in range(g.number_of_edges()):
                g.edges[edge_list[i]]['attributes'] = edge_attributes[i]
                i += 1

    # Classes.
    if path.exists("./datasets/" + pre + ds_name + "/" + ds_name + "/" + "raw/" + ds_name + "_graph_labels.txt"):
        with open("./datasets/" + pre + ds_name + "/" + ds_name + "/" + "raw/" + ds_name + "_graph_labels.txt",
                  "r") as f:
            classes = [map(int, i.split(',')) for i in list(f)]  # [int(i) for i in list(f)]
        f.closed
        i = 0
        for g in graph_db:
            g.graph['classes'] = classes[i]
            i += 1

    # Targets.
    if path.exists("./datasets/" + pre + ds_name + "/" + ds_name + "/" + "raw/" + ds_name + "_graph_attributes.txt"):
        with open("./datasets/" + pre + ds_name + "/" + ds_name + "/" + "raw/" + ds_name + "_graph_attributes.txt",
                  "r") as f:
            targets = [map(float, i.split(',')) for i in list(f)]
        f.closed
        i = 0
        for g in graph_db:
            g.graph['targets'] = targets[i]
            i += 1

    return graph_db
