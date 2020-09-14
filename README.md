# TUDataset

Source code for our ICML 2020 Workshop on Graph Representation Learning and Beyond (GRL+ 2020) paper "[TUDataset: A collection of benchmark datasets for learning with graphs](https://grlplus.github.io/papers/79.pdf)". This repository contains graph kernel and GNN baseline implementations, data loaders, and evaluations scripts.

See [graphlearning.io](http://www.graphlearning.io/) for documentation on how to use this package.




## Installation

First make sure that you have all requirements installed.
###  Requirements
- `networkx`
- `numpy` (somewhat recent version)
- `pandas` (somewhat recent version)
- `Python` 3.x
- `scipy` (somewhat recent version)
- `sklearn` (somewhat recent version)
- `torch` 1.5
- `torch-geometric` 1.5

If you want to use the kernel baselines, you will further need:
- `eigen3`
- `g++` 
- `pybind11`

### Compilation of kernel baselines
To execute the kernel baselines you have to compile the Python package. If you just want to use the GNN baselines, data loaders, and evaluation scripts, you can skip this step.

Execute the following steps: 
```Bash
$ cd tud_benchmark/kernel_baselines
```
If you are using a Linux system, run
```Bash
$ g++ -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 --includes`  kernel_baselines.cpp src/*cpp -o ../kernel_baselines`python3-config --extension-suffix`
```
on MacOS, run
```Bash
$ g++ -O3 -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes`  kernel_baselines.cpp src/*cpp -o ../kernel_baselines`python3-config --extension-suffix`
```

You might need to adjust your path to ``pybind11`` and ``eigen3`` in ``kernel_baselines.cpp``, 
``kernel_baselines/src/AuxiliaryMethods.h``, and ``kernel_baselines/src/Graph.cpp``. 

## Usage
See ``tud_benchmark/main_kernel.py`` and ``tud_benchmark/main_gnn.py`` for baseline and evaluation examples. More details can be found at [graphlearning.io](https://chrsmrrs.github.io/datasets/docs/evaluation/).

## Terms and conditions
Please feel free to use our code. We only ask that you cite:

```
@inproceedings{Morris+2020,
    title={TUDataset: A collection of benchmark datasets for learning with graphs},
    author={Christopher Morris and Nils M. Kriege and Franka Bause and Kristian Kersting and Petra Mutzel and Marion Neumann},
    booktitle={ICML 2020 Workshop on Graph Representation Learning and Beyond (GRL+ 2020)},
    pages={},
    url={www.graphlearning.io}
    year={2020}
}
```

## Contact Information
If you have any questions, send an email to Christopher Morris (christopher.morris at tu-dortmund.de).
