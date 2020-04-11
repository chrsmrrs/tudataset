/**********************************************************************
 * Copyright (C) 2017 Christopher Morris <christopher.morris@udo.edu>
 *********************************************************************/

#include <cstdio>
#include "src/AuxiliaryMethods.h"

#include "src/ColorRefinementKernel.h"
#include "src/GenerateTwo.h"
#include "src/GraphletKernel.h"
#include "src/ShortestPathKernel.h"
#include "src/GenerateTwo.h"


#include "src/Graph.h"
#include <iostream>
#include <chrono>


#ifdef __linux__
    #include <pybind11/pybind11.h>
    #include <pybind11/eigen.h>
    #include <pybind11/stl.h>
#else
    #include </usr/local/include/pybind11/pybind11.h>
    #include </usr/local/include/pybind11/stl.h>
    #include </usr/local/include/pybind11/eigen.h>
#endif


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
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, true, false);

    return MatrixXd(gm);
}

GramMatrix compute_lwl_2_sparse_ZINC(int num_iterations,  bool use_labels,  bool use_edge_labels, const std::vector<int> &indices_train, const std::vector<int> &indices_val, const std::vector<int> &indices_test) {

    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

   cout << "$$$" << endl;
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb.erase(gdb.begin() + 0);
    cout << "$$$" << endl;
        GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);
   cout << "$$$" << endl;


   GraphDatabase gdb_new;
   for (auto i : indices_train) {
       gdb_new.push_back(gdb[int(i)]);
   }
   cout << gdb_new.size() << endl;
   cout << "$$$" << endl;

   for (auto i : indices_val) {
       gdb_new.push_back(gdb_2[int(i)]);
   }
   cout << gdb_new.size() << endl;
   cout << "$$$" << endl;


   for (auto i : indices_test) {
       gdb_new.push_back(gdb_3[int(i)]);
   }
   cout << gdb_new.size() << endl;
   cout << "$$$" << endl;


    GenerateTwo::GenerateTwo wl(gdb_new);
    GramMatrix gm;
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, "local", false, false, false);

   cout << "$$$" << endl;

    return gm;
}


vector<GramMatrix> compute_wl_1_sparse_ZINC(bool use_labels,  bool use_edge_labels, const std::vector<int> &indices_train, const std::vector<int> &indices_val, const std::vector<int> &indices_test) {

    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

   cout << "$$$" << endl;
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb.erase(gdb.begin() + 0);
    cout << "$$$" << endl;
        GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);
   cout << "$$$" << endl;


   GraphDatabase gdb_new;
   for (auto i : indices_train) {
       gdb_new.push_back(gdb[int(i)]);
   }
   cout << gdb_new.size() << endl;
   cout << "$$$" << endl;

   for (auto i : indices_val) {
       gdb_new.push_back(gdb_2[int(i)]);
   }
   cout << gdb_new.size() << endl;
   cout << "$$$" << endl;


   for (auto i : indices_test) {
       gdb_new.push_back(gdb_3[int(i)]);
   }
   cout << gdb_new.size() << endl;
   cout << "$$$" << endl;


    ColorRefinement::ColorRefinementKernel wl(gdb_new);
    GramMatrix gm;
    vector<GramMatrix> matrices

    for (int i = 0; i < 6; ++i) {
         cout << i << endl;
         gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, false, false);
         matrices.push_back(gm);
    }


    return matrices;
}



GramMatrix compute_wl_2_sparse_ZINC(int num_iterations,  bool use_labels,  bool use_edge_labels, const std::vector<int> &indices_train, const std::vector<int> &indices_val, const std::vector<int> &indices_test) {

    GraphDatabase gdb_2 = AuxiliaryMethods::read_graph_txt_file("ZINC_val");
    gdb_2.erase(gdb_2.begin() + 0);

   cout << "$$$" << endl;
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file("ZINC_train");
    gdb.erase(gdb.begin() + 0);
    cout << "$$$" << endl;
        GraphDatabase gdb_3 = AuxiliaryMethods::read_graph_txt_file("ZINC_test");
    gdb_3.erase(gdb_3.begin() + 0);
   cout << "$$$" << endl;


   GraphDatabase gdb_new;
   for (auto i : indices_train) {
       gdb_new.push_back(gdb[int(i)]);
   }
   cout << gdb_new.size() << endl;
   cout << "$$$" << endl;

   for (auto i : indices_val) {
       gdb_new.push_back(gdb_2[int(i)]);
   }
   cout << gdb_new.size() << endl;
   cout << "$$$" << endl;


   for (auto i : indices_test) {
       gdb_new.push_back(gdb_3[int(i)]);
   }
   cout << gdb_new.size() << endl;
   cout << "$$$" << endl;


    ColorRefinement::ColorRefinementKernel wl(gdb_new);
    GramMatrix gm;
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, false, false);

    return gm;
}


MatrixXd compute_wloa_dense(string ds, int num_iterations,  bool use_labels,  bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);

    ColorRefinement::ColorRefinementKernel wl(gdb);
    GramMatrix gm;
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, true, true);

    return MatrixXd(gm);
}


GramMatrix compute_wl_1_sparse(string ds, int num_iterations, bool use_labels, bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);

    ColorRefinement::ColorRefinementKernel wl(gdb);
    GramMatrix gm;
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, false, false);

    return gm;
}

MatrixXd compute_lwl_2_dense(string ds, int num_iterations,  bool use_labels,  bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);

    GenerateTwo::GenerateTwo wl(gdb);
    GramMatrix gm;
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, "local", false, true, false);

    return MatrixXd(gm);
}

MatrixXd compute_lwl_2_wloa_dense(string ds, int num_iterations,  bool use_labels,  bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);

    GenerateTwo::GenerateTwo wl(gdb);
    GramMatrix gm;
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, "localp", false, true, true);

    return MatrixXd(gm);
}

MatrixXd compute_lwlp_2_wloa_dense(string ds, int num_iterations,  bool use_labels,  bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);

    GenerateTwo::GenerateTwo wl(gdb);
    GramMatrix gm;
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, "localp", false, true, true);

    return MatrixXd(gm);
}



GramMatrix compute_lwl_2_sparse(string ds, int num_iterations, bool use_labels,  bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);

    GenerateTwo::GenerateTwo wl(gdb);
    GramMatrix gm;
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, "local", false, false, false);

    return gm;
}

MatrixXd compute_lwlp_2_dense(string ds, int num_iterations,  bool use_labels,  bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);

    GenerateTwo::GenerateTwo wl(gdb);
    GramMatrix gm;
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, "localp", false, true, false);

    return MatrixXd(gm);
}

GramMatrix compute_lwlp_2_sparse(string ds, int num_iterations, bool use_labels,  bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);

    GenerateTwo::GenerateTwo wl(gdb);
    GramMatrix gm;
    gm = wl.compute_gram_matrix(num_iterations, use_labels, use_edge_labels, "localp", false, true, false);

    return gm;
}

GramMatrix compute_graphlet_sparse(string ds, bool use_labels, bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);
    vector<int> classes = AuxiliaryMethods::read_classes(ds);

    GraphletKernel::GraphletKernel graphlet(gdb);
    GramMatrix gm;
    gm = graphlet.compute_gram_matrix(use_labels, use_edge_labels, false);

    return gm;
}

MatrixXd compute_graphlet_dense(string ds, bool use_labels, bool use_edge_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);
    vector<int> classes = AuxiliaryMethods::read_classes(ds);

    GraphletKernel::GraphletKernel graphlet(gdb);
    GramMatrix gm;
    gm = graphlet.compute_gram_matrix(use_labels, use_edge_labels, true);

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

GramMatrix compute_shortestpath_sparse(string ds, bool use_labels) {
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
    gdb.erase(gdb.begin() + 0);
    vector<int> classes = AuxiliaryMethods::read_classes(ds);

    ShortestPathKernel::ShortestPathKernel sp(gdb);
    GramMatrix gm;
    gm = sp.compute_gram_matrix(use_labels, false);

    return gm;
}


vector<float> read_targets(string data_set_name, const std::vector<int> &indices) {

   vector<float> targets =  AuxiliaryMethods::read_targets(data_set_name);

   vector<float> new_targets;
   for (auto i : indices) {
       new_targets.push_back(targets[i]);
   }

  return new_targets;

}

PYBIND11_MODULE(kernel_baselines, m) {
    m.def("compute_wl_1_dense", &compute_wl_1_dense);
    m.def("compute_wloa_dense", &compute_wloa_dense);
    m.def("compute_wl_1_sparse", &compute_wl_1_sparse);
    m.def("compute_wl_1_sparse_ZINC", &compute_wl_1_sparse_ZINC);
    m.def("compute_lwl_2_sparse_ZINC", &compute_lwl_2_sparse_ZINC);

    m.def("read_targets", &read_targets);



    m.def("compute_lwl_2_dense", &compute_lwl_2_dense);
    m.def("compute_lwl_2_sparse", &compute_lwl_2_sparse);
    m.def("compute_lwl_2_wloa_dense", &compute_lwl_2_wloa_dense);
    m.def("compute_lwlp_2_wloa_dense", &compute_lwlp_2_wloa_dense);

    m.def("compute_lwlp_2_dense", &compute_lwlp_2_dense);
    m.def("compute_lwlp_2_sparse", &compute_lwlp_2_sparse);

    m.def("compute_graphlet_dense", &compute_graphlet_dense);
    m.def("compute_graphlet_sparse", &compute_graphlet_sparse);
    m.def("compute_shortestpath_dense", &compute_shortestpath_dense);
    m.def("compute_shortestpath_sparse", &compute_shortestpath_sparse);
}
