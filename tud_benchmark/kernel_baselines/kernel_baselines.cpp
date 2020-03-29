/**********************************************************************
 * Copyright (C) 2017 Christopher Morris <christopher.morris@udo.edu>
 *********************************************************************/

#include <cstdio>
#include "src/AuxiliaryMethods.h"

#include "src/ColorRefinementKernel.h"
#include "src/GraphletKernel.h"
#include "src/ShortestPathKernel.h"
#include "src/GenerateTwo.h"

#include "src/Graph.h"
#include <iostream>
#include <chrono>


#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
//#include </usr/local/include/pybind11/pybind11.h>
//#include </usr/local/include/pybind11/eigen.h>



namespace py = pybind11;
using namespace std::chrono;
using namespace std;
using namespace GraphLibrary;

using namespace std;

MatrixXd compute_wl_1_dense(string ds, int num_iterations,  bool use_labels,  bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);

    ColorRefinement::ColorRefinementKernel wl(gdb);
    GramMatrix gm;
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, true);

    return MatrixXd(gm);
}

GramMatrix compute_wl_1_sparse(string ds, int num_iterations, bool use_labels, bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);

    ColorRefinement::ColorRefinementKernel wl(gdb);
    GramMatrix gm;
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, true);

    return gm;
}

MatrixXd compute_lwl_2_dense(string ds, int num_iterations,  bool use_labels,  bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);

    GenerateTwo::GenerateTwo wl(gdb);
    GramMatrix gm;
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, "local", false, true);

    return MatrixXd(gm);
}

GramMatrix compute_lwl_2_sparse(string ds, int num_iterations, bool use_labels,  bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);

    GenerateTwo::GenerateTwo wl(gdb);
    GramMatrix gm;
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, "local", false, true);

    return gm;
}

MatrixXd compute_lwlp_2_dense(string ds, int num_iterations,  bool use_labels,  bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);

    GenerateTwo::GenerateTwo wl(gdb);
    GramMatrix gm;
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, "localp", false, true);

    return MatrixXd(gm);
}

GramMatrix compute_lwlp_2_sparse(string ds, int num_iterations, bool use_labels,  bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);

    GenerateTwo::GenerateTwo wl(gdb);
    GramMatrix gm;
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, "localp", false, true);

    return gm;
}


GramMatrix compute_graphlet_sparse(string ds, bool use_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);
    vector<int> classes = AuxiliaryMethods::read_classes(ds);

    GraphletKernel::GraphletKernel graphlet(gdb);
    GramMatrix gm;
    gm = graphlet.compute_gram_matrix(use_labels, false);

    return gm;
}

MatrixXd compute_graphlet_dense(string ds, bool use_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);
    vector<int> classes = AuxiliaryMethods::read_classes(ds);

    GraphletKernel::GraphletKernel graphlet(gdb);
    GramMatrix gm;
    gm = graphlet.compute_gram_matrix(use_labels, true);

    return MatrixXd(gm);
}

MatrixXd compute_shortestpath_dense(string ds, bool use_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);
    vector<int> classes = AuxiliaryMethods::read_classes(ds);

    ShortestPathKernel::ShortestPathKernel sp(gdb);
    GramMatrix gm;
    gm = sp.compute_gram_matrix(use_labels, true);

    return MatrixXd(gm);
}




PYBIND11_MODULE(kernel_baselines, m) {
    m.def("compute_wl_1_dense", &compute_wl_1_dense);
    m.def("compute_wl_1_sparse", &compute_wl_1_sparse);

    m.def("compute_lwl_2_dense", &compute_lwl_2_dense);
    m.def("compute_lwl_2_sparse", &compute_lwl_2_sparse);

    m.def("compute_lwlp_2_dense", &compute_lwlp_2_dense);
    m.def("compute_lwlp_2_sparse", &compute_lwlp_2_sparse);

    m.def("compute_graphlet_dense", &compute_graphlet_dense);
    m.def("compute_graphlet_sparse", &compute_graphlet_sparse);
    m.def("compute_shortestpath_dense", &compute_shortestpath_dense);
}
