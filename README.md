# Digital Annealer On EE Server
This repo contains code to find the configuration of the spins such that Ising Hamiltonian is minimum.

## Code Structure
This repo contains 

	- annealer_gpu_SI
	- bin_SI
	- build(generated later)
	- anneal.cmd
		
executable has cmdline feature to provide different options for help cmd use "annealer_gpu_SI -h". In debug mode(use "-d") executable will generate average magnetization(avg_magnet_XYZ.txt) for a temperature and final spin configuration(spins_XYZ.txt) in build folder if the <b>J_matrix's name ends with "_XYZ.txt" </b>

## Build the project


* First copy the J_matrix text file in bin_SI folder

* Run "chmod +x script.sh"

* Run "./script.sh" in the terminal

* Go to bin_SI 

* Run executable using this cmd

	- "./annealer_gpu_SI -a ../bin_SI/J_Matrix_40x40.txt  -l ../bin_SI/linear_4x8.txt -x 6.4 -y 0.001 -n 35 -m 9000 -d"

	- here -a opt for passing J_matrix text file(makes sure to have _ in your file name "_40x40.txt")
	- -l opt for passing linear values text file(makes sure to have _ in your file name "_40x40.txt")
	- -x starting temperature and -y is the final temperature
	- -n opt for number of temperatures in the provided range
	- -m opt for number iteration at a given temperature
	- -d opt for generating files contains final spin configuration and avg_magnetism with total cut value.



	- for example if ur J_matrix is abc_XYZ.txt use the cmdline option "-a ../bin_SI/abc_XYZ.txt -d" then the debug files are generated in same folder with name ending with _XYZ.txt

	- Another example "annealer_gpu_SI -a ../bin_SI/J_Matrix_40x40.txt -x 6.4 -y 0.001 -n 35 -m 9000 -d"

## File Format for Coupling constants and linear terms
There are 2 files needed to run the Ising solver. One is the coupling constants J_matrix_XxY.txt and linear/bias terms in linear_XxY.txt. The term Jij terms are provided in the following format:

```
        n   m
        i_1 j_1 J_{i_1,j_1}
        i_2 j_2 J_{i_2,j_2}
        ...
        i_m j_m J_{i_m,j_m}
```
where n is the number of variables/vertices, m the number of entries/nodes of J specified in the subsequent list. J is assumed to be symmetric and indexed by numbers from 1 to n. Hence, if you specify entry (i,j) having value J_ij, also entry (j,i) has value J_ij. The values of J can be integers or reals.

For linear terms(i.e. h_i) are provided in the following format:

```
h_1 h_2 ..... h_n
```
all the h_i terms are separated by single space. For more information please look in reference folder.

# Acknowledgement
Thanks to Dr. Anil Prabhakar and Dr. Nitin Chandrachoodan for spending valuable time to guide me. I thank Dhruv for his valubale information for conversion of QUBO to Ising model.
