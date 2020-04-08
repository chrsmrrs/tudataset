import numpy as np
import networkx as nx
import os.path as path
from torch_geometric.datasets import TUDataset
import os.path as osp



# Return classes as numpy array.
def read_classes(ds_name):
    # Classes
    with open("datasets/" +  ds_name + "/raw/" + ds_name + "_graph_labels.txt", "r") as f:
        classes = [int(i) for i in list(f)]
    f.closed

    return np.array(classes)

def read_targets(ds_name):
    # Classes
    with open("datasets/" + ds_name  + "/" + ds_name  + "/raw/" + ds_name + "_graph_attributes.txt", "r") as f:
        classes = [float(i) for i in list(f)]
    f.closed

    return np.array(classes)


def get_dataset(dataset, regression=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', dataset)
    TUDataset(path, name=dataset)

    if not regression:
        return read_classes(dataset)
    else:
        return read_targets(dataset)

# TODO: Finsih this.
# Return dataset as list of networkx graphs.
def read_graphs_nx(ds_name):
    pre = ""

    with open("datasets/" + pre + ds_name + "/" + ds_name + "_graph_indicator.txt", "r") as f:
        graph_indicator = [int(i) - 1 for i in list(f)]
    f.closed

    # Nodes
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
    vertex_list = []
    for i in node_indices:
        # Undirected graph.
        g = nx.Graph()
        vertex_list_g = []

        for n, _ in enumerate(range(i[1] - i[0] + 1)):
            vertex_list_g.append(n)
            g.add_node(n)

        graph_db.append(g)
        vertex_list.append(vertex_list_g)

    # Edges
    with open("datasets/" + pre + ds_name + "/" + ds_name + "_A.txt", "r") as f:
        edges = [i.split(',') for i in list(f)]
    f.closed

    edges = [(int(e[0].strip()) - 1, int(e[1].strip()) - 1) for e in edges]

    edge_indicator = []
    edge_list = []
    i = 0
    for e in edges:
        g_id = graph_indicator[e[0]]
        edge_indicator.append(g_id)
        g = graph_db[g_id]
        off = offset[g_id]

        # Avoid multigraph
        if not g.has_edge(e[0] - off, e[1] - off):
            g.add_edge(e[0] - off, e[1] - off)
            edge_list.append(i)
            i += 1
    # Node labels
    if path.exists("datasets/" + pre + ds_name + "/" + ds_name + "_node_labels.txt"):
        with open("datasets/" + pre + ds_name + "/" + ds_name + "_node_labels.txt", "r") as f:
            node_labels = [int(i) for i in list(f)]
        f.closed

        for g_id, g in enumerate(graph_db):
            off = offset[g_id]
            nx.set_node_attributes(g, name='node_labels', values=listToDict(node_labels[off:off + g.number_of_nodes()]))

    # Node Attributes
    if path.exists("datasets/" + pre + ds_name + "/" + ds_name + "_node_attributes.txt"):
        with open("datasets/" + pre + ds_name + "/" + ds_name + "_node_attributes.txt", "r") as f:
            node_attributes = [np.array([float(j) for j in i.split(',')]) for i in list(f)]
        f.closed

        for g_id, g in enumerate(graph_db):
            off = offset[g_id]
            nx.set_node_attributes(g, name='node_attributes', values=listToDict(node_attributes[off:off + g.number_of_nodes()]))


    # Edge Labels
    if path.exists("datasets/" + ds_name + "/" + ds_name + "_edge_labels.txt"):
        with open("datasets/" + ds_name + "/" + ds_name + "_edge_labels.txt", "r") as f:
            edge_labels = [int(i) for i in list(f)]
        f.closed

        l_el = []
        for i in range(num_graphs + 1):
            g = graph_db[graph_indicator[i]]
            l_el.append(g.new_edge_property("int"))

        for i, l in enumerate(edge_labels):
            g_id = edge_indicator[i]
            g = graph_db[g_id]

            l_el[g_id][edge_list[i]] = l
            g.ep.el = l_el[g_id]

    exit()

    # Edge Attributes
    if path.exists("datasets/" + ds_name + "/" + ds_name + "_edge_attributes.txt"):
        with open("datasets/" + ds_name + "/" + ds_name + "_edge_attributes.txt", "r") as f:
            edge_attributes = [map(float, i.split(',')) for i in list(f)]
        f.closed

        l_ea = []
        for i in range(num_graphs + 1):
            g = graph_db[graph_indicator[i]]
            l_ea.append(g.new_edge_property("vector<float>"))

        for i, l in enumerate(edge_attributes):
            g_id = edge_indicator[i]
            g = graph_db[g_id]

            l_ea[g_id][edge_list[i]] = l
            g.ep.ea = l_ea[g_id]

    # Classes
    with open("datasets/" + pre + ds_name + "/" + ds_name + "_graph_labels.txt", "r") as f:
        classes = np.array([int(i) for i in list(f)])
    f.closed

    return graph_db, classes