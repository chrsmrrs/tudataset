/**********************************************************************
 * Copyright (C) 2020 Christopher Morris <christopher.morris@udo.edu>
 *********************************************************************/

#include "AuxiliaryMethods.h"

using Eigen::IOFormat;
using Eigen::MatrixXd;
using namespace std;

namespace AuxiliaryMethods {
    vector<int> split_string(string s) {
        vector<int> result;
        stringstream ss(s);

        while (ss.good()) {
            string substr;
            getline(ss, substr, ',');
            result.push_back(stoi(substr));
        }

        return result;
    }

    GraphDatabase read_graph_txt_file(string data_set_name) {
        string line;

        string path = ".";

        vector<uint> graph_indicator;
        ifstream myfile(
                path + "/datasets/"+ data_set_name +  "/"  + data_set_name +  "/raw/" + data_set_name + "_graph_indicator.txt");
        if (myfile.is_open()) {
            while (getline(myfile, line)) {
                graph_indicator.push_back(stoi(line));
            }
            myfile.close();
        } else {
            printf("%s", "!!! Unable to open file 1 !!!\n");
            exit(EXIT_FAILURE);
        }

        uint num_graphs = graph_indicator.back() + 1;

        // Get labels from for each node.
        bool label_data = true;
        string label;
        Labels node_labels;
        ifstream labels(
                path + "/datasets/"+ data_set_name +  "/"  + data_set_name + "/raw/" + data_set_name + "_node_labels.txt");
        if (labels.is_open()) {
            while (getline(labels, label)) {
                node_labels.push_back(stoul(label));
            }
            myfile.close();
        } else {
            label_data = false;
        }

        GraphDatabase graph_database;
        unordered_map<int, int> offset;
        int num_nodes = 0;

        // Add vertices to each graph in graph database and assign labels.
        for (uint i = 0; i < num_graphs; ++i) {
            pair<int, int> p(i, num_nodes);
            offset.insert(p);
            unsigned long s = count(graph_indicator.begin(), graph_indicator.end(), i);

            Labels l;
            if (label_data) {
                for (unsigned long j = num_nodes; j < s + num_nodes; ++j) {
                    l.push_back(node_labels[j]);
                }
            }

            num_nodes += s;
            EdgeList edge_list;

            Graph new_graph(false, s, edge_list, l);
            graph_database.push_back(new_graph);
        }


        // Get labels from for each node.
        bool edge_label_data = true;
        Labels edge_labels;
        ifstream elabels(path + "/datasets/"+ data_set_name +  "/"  + data_set_name + "/raw/" + data_set_name + "_edge_labels.txt");
        if (elabels.is_open()) {
            while (getline(elabels, label)) {
                edge_labels.push_back(stoul(label));
            }
            myfile.close();
        } else {
            edge_label_data = false;
        }

        // Insert edges for each graph.
        vector<EdgeLabels> edge_label_vector;
        for (uint i = 0; i < num_graphs; ++i) {
            edge_label_vector.push_back(EdgeLabels());
        }


        uint c = 0;
        vector<int> edges;
        ifstream edge_file(path + "/datasets/" + data_set_name +  "/" + data_set_name + "/raw/" + data_set_name + "_A.txt");
        if (edge_file.is_open()) {
            while (getline(edge_file, line)) {
                vector<int> r = split_string(line);

                uint graph_num = graph_indicator[r[0] - 1];
                uint off = offset[graph_num];
                Node v = r[0] - 1 - off;
                Node w = r[1] - 1 - off;

                if (!graph_database[graph_num].has_edge(v, w)) {
                    graph_database[graph_num].add_edge(v, w);
                }

                if (edge_label_data) {
                    edge_label_vector[graph_num].insert({{make_tuple(v, w), edge_labels[c]}});
                    edge_label_vector[graph_num].insert({{make_tuple(w, v), edge_labels[c]}});
                }

                edges.push_back(stoi(line));
                c++;

            }
            edge_file.close();
        } else {
            printf("%s", "!!! Unable to open file 2!!!\n");
            exit(EXIT_FAILURE);
        }

        if (edge_label_data) {
            for (uint i = 0; i < num_graphs; ++i) {
                graph_database[i].set_edge_labels(edge_label_vector[i]);
            }
        }

        return graph_database;
    }

    vector<int> read_classes(string data_set_name) {
        string line;

        string path = ".";
        //string path = "/Users/chrsmrrs/localwl_dev";
        vector<int> classes;

        ifstream myfile(
                path + "/datasets/"+ data_set_name +  "/"  + data_set_name  +"/raw/" + data_set_name +
                "_graph_labels.txt");
        if (myfile.is_open()) {
            while (getline(myfile, line)) {
                classes.push_back(stoi(line));
            }
            myfile.close();
        } else {
            printf("%s", "!!! Unable to open file !!!\n");
            exit(EXIT_FAILURE);
        }

        return classes;
    }

    vector<float> read_targets(string data_set_name) {
        string line;

        string path = ".";
        //string path = "/Users/chrsmrrs/localwl_dev";
        vector<float> classes;

        ifstream myfile(
                path + "/datasets/"+ data_set_name +  "/" + data_set_name + "/raw/" + data_set_name +
                "_graph_attributes.txt");
        if (myfile.is_open()) {
            while (getline(myfile, line)) {
                classes.push_back(stof(line));
            }
            myfile.close();
        } else {
            printf("%s", "!!! Unable to open file !!!\n");
            exit(EXIT_FAILURE);
        }

        return classes;
    }

    Label pairing(const Label a, const Label b) {
        return a >= b ? a * a + a + b : a + b * b;
    }
    
    Label pairing(const vector<Label> labels) {
        Label new_label=labels.size();
        for (Label l: labels) {
            new_label = pairing(new_label, l);
        }
        return new_label;
    }
}
