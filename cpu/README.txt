# Building for source

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Source files requires g++ compiler to build

```sh
g++ utils.hpp ising_cpu.cpp -o ising_cpu
```

To run the code provide the following information
Start temperature
Number of temperature points from start temperature to end temperature(i.e. 0.01f)
Number of iteration in a particular temperature
Coupling constant filename

# Run

```sh
./ising_cpu 15.8 35 5000 J_Matrix_20x20.txt
```