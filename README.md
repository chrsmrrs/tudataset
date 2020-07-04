# TUDataset

Source code for our ICML 2020 Workshop on Graph Representation Learning and Beyond (GRL+ 2020) paper "[TUDataset: A collection of benchmark datasets for learning with graphs](https://arxiv.org/abs/XXX.XXX)". This repository contains graph kernel and GNN baseline implementations, data loaders, and evaluations scripts.

See [graphlearning.io](http://www.graphlearning.io/) for documentation on how to use this package.




## Installation

First make sure that you have all requirements installed.
###  Requirements
- Python 3.8
- eigen3
- numpy
- pandas
- scipy
- sklearn
- torch 1.5
- torch-geometric
- pybind11
- libsvm




Using `cmake` you can simply type `cmake cmake-build-debug`, otherwise (using `gcc`)

```Bash
$ g++ main.cpp src/*.h src/*.cpp -std=c++11 -o wlglobal -O2
```
In order to compile, you need a recent version of [Eigen 3](http://eigen.tuxfamily.org/index.php?title=Main_Page) installed on your system.

## Terms and conditions
Please feel free to use our code. We only ask that you cite:

	@inproceedings{Morris+2020,
	    title={TUDataset: A collection of benchmark datasets for learning with graphs},
	    author={Christopher Morris, Nils M. Kriege, Franka Bause, Kristian Kersting, Petra Mutzel, Marion Neumann},
	    booktitle={ICML 2020 Workshop on Graph Representation Learning and Beyond (GRL+ 2020)},
	    pages={},
	    year={2020}
	}

## Contact Information
If you have any questions, send an email to Christopher Morris (christopher.morris at tu-dortmund.de).
