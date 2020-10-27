import os.path as path
import networkx as nx


def tud_to_networkx(ds_name):
    pre = ds_name + "/"

    with open("./datasets/" + pre + ds_name + "/raw/" + ds_name + "_graph_indicator.txt", "r") as f:
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
        for j in range(i[1] - i[0]+1):
            g.add_node(j)

        graph_db.append(g)

    # Edges.
    with open("./datasets/" + pre + ds_name + "/raw/" + ds_name + "_A.txt", "r") as f:
        edges = [i.split(',') for i in list(f)]
    f.closed
    
    edges = [(int(e[0].strip()) - 1, int(e[1].strip()) - 1) for e in edges]
    edge_list = []
    edgeb_list = []
    for e in edges:
        g_id = graph_indicator[e[0]]
        g = graph_db[g_id]
        off = offset[g_id]

        # Avoid multigraph (for edge_list)
        if ((e[0] - off, e[1] - off) not in list(g.edges())) and ((e[1] - off, e[0] - off) not in list(g.edges())):
            g.add_edge(e[0] - off, e[1] - off)
            edge_list.append((e[0] - off, e[1] - off))
            edgeb_list.append(True)
        else:
            edgeb_list.append(False)

    # Node labels.
    if path.exists("./datasets/" + pre + ds_name + "/raw/" + ds_name + "_node_labels.txt"):
        with open("./datasets/" + pre + ds_name + "/raw/" + ds_name + "_node_labels.txt", "r") as f:
            node_labels = [str.strip(i) for i in list(f)]
        f.closed
        
        node_labels = [i.split(',') for i in node_labels]
        int_labels = [];
        for i in range(len(node_labels)):
            int_labels.append([int(j) for j in node_labels[i]])
        
        i = 0
        for g in graph_db:
            for v in range(g.number_of_nodes()):
                g.nodes[v]['labels'] = int_labels[i]
                i += 1

    # Node Attributes.
    if path.exists("./datasets/" + pre + ds_name + "/raw/" + ds_name + "_node_attributes.txt"):
        with open("./datasets/" + pre + ds_name + "/raw/" + ds_name + "_node_attributes.txt", "r") as f:
            node_attributes = [str.strip(i) for i in list(f)]
        f.closed
        
        node_attributes = [i.split(',') for i in node_attributes]
        float_attributes = [];
        for i in range(len(node_attributes)):
            float_attributes.append([float(j) for j in node_attributes[i]])
        i = 0
        for g in graph_db:
            for v in range(g.number_of_nodes()):
                g.nodes[v]['attributes'] = float_attributes[i]
                i += 1

    # Edge Labels.
    if path.exists("./datasets/" + pre + ds_name + "/raw/" + ds_name + "_edge_labels.txt"):
        with open("./datasets/" + pre + ds_name + "/raw/" + ds_name + "_edge_labels.txt", "r") as f:
            edge_labels = [str.strip(i) for i in list(f)]
        f.closed

        edge_labels = [i.split(',') for i in edge_labels]
        e_labels = [];
        for i in range(len(edge_labels)):
            if(edgeb_list[i]):
                e_labels.append([int(j) for j in edge_labels[i]])
        
        i = 0
        for g in graph_db:
            for e in range(g.number_of_edges()):
                g.edges[edge_list[i]]['labels'] = e_labels[i]
                i += 1

    # Edge Attributes.
    if path.exists("./datasets/" + pre + ds_name + "/raw/" + ds_name + "_edge_attributes.txt"):
        with open("./datasets/" + pre + ds_name + "/raw/" + ds_name + "_edge_attributes.txt", "r") as f:
            edge_attributes = [str.strip(i) for i in list(f)]
        f.closed

        edge_attributes = [i.split(',') for i in edge_attributes]
        e_attributes = [];
        for i in range(len(edge_attributes)):
            if(edgeb_list[i]):
                e_attributes.append([float(j) for j in edge_attributes[i]])
        
        i = 0
        for g in graph_db:
            for e in range(g.number_of_edges()):
                g.edges[edge_list[i]]['attributes'] = e_attributes[i]
                i += 1

    # Classes.
    if path.exists("./datasets/" + pre + ds_name + "/raw/" + ds_name + "_graph_labels.txt"):
        with open("./datasets/" + pre + ds_name + "/raw/" + ds_name + "_graph_labels.txt", "r") as f:
            classes = [str.strip(i) for i in list(f)]
        f.closed
        classes = [i.split(',') for i in classes]
        cs = [];
        for i in range(len(classes)):
            cs.append([int(j) for j in classes[i]])
        
        i = 0
        for g in graph_db:
            g.graph['classes'] = cs[i]
            i += 1

    # Targets.
    if path.exists("./datasets/" + pre + ds_name + "/raw/" + ds_name + "_graph_attributes.txt"):
        with open("./datasets/" + pre + ds_name + "/raw/" + ds_name + "_graph_attributes.txt", "r") as f:
            targets = [str.strip(i) for i in list(f)]
        f.closed
        
        targets= [i.split(',') for i in targets]
        ts = [];
        for i in range(len(targets)):
            ts.append([float(j) for j in targets[i]])
        
        i = 0
        for g in graph_db:
            g.graph['targets'] = ts[i]
            i += 1

    return graph_db
