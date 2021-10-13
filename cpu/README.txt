# Build 

Single source file, requires g++ compiler to build

“g++ ising_cpu.cc -o ising_cpu”

To run the code provide the following information
Start temperature
Number of temperature points from start temperature to end temperature(i.e. 0.01f)
Number of iteration in a particular temperature
Coupling constant filename
Linear constant filename

# Run

For example :
          ./ising_cpu 3.26 35 1000 adj_mat_20x20.txt linear_20x20.txt
