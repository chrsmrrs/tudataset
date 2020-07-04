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
- g++ 

### Compilation`
In order to execute the kernel baseline you have to compile the Python package. If you just want to use the GNN baseline and evaluation scripts, you can skip this step.

Execute the following steps: 
```Bash
$ cd tud_benchmark/kernel_baselines
```
If you are using a Linux system, run
```Bash
$ g++ -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 --includes`  kernel_baselines.cpp src/*cpp -o ../kernel_baselines`python3-config --extension-suffix`
```
on MacOs, run
```Bash
$ g++ -O3 -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes`  kernel_baselines.cpp src/*cpp -o ../kernel_baselines`python3-config --extension-suffix`
```

You might need to adjust your path to ``pybind11`` and ``eigen3`` in ``kernel_baselines.cpp``, 
``src/AuxiliaryMethods.h``, and ``Graph.cpp``. 


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
