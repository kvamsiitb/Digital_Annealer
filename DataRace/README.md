# Build

Source files require a C++ compiler to build.

```sh
g++ utils.hpp ising_cpu.cpp -o ising_cpu
```

# Run

To run the code provide the following information:
- Start temperature
- Number of temperature points from start temperature to end temperature (i.e. `0.01f`)
- Number of iterations at a particular temperature
- Coupling constant filename
- Linear constant filename

```sh
./ising_cpu 15.8 35 5000 constants/J_Matrix_20x20.txt constants/linear_20x20.txt
```