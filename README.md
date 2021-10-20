# Digital Annealer On EE Server
This repo contains code to find the configuration of the spins such that Ising Hamiltonian is minimum.

## Monte Carlo Metropolis Method


## Simulated Annealing


## Code Structure
This repo contains 

	- annealer_gpu_SI
	- bin_SI
	- build(generated later)
	- anneal.cmd
		
executable has cmdline feature to provide different options for help cmd use "annealer_gpu_SI -h". In debug mode(use "-d") executable will generate average magnetization(avg_magnet_XYZ.txt) for a temperature and final spin configuration(spins_XYZ.txt) in build folder if the <b>J_matrix's name ends with "_XYZ.txt" <\b>

## Build the project


* First copy the J_matrix text file in bin_SI folder

* Run "chmod +x script.sh"

* Run "./script.sh" in the terminal

* Go to bin_SI 

* Run executable using this cmd

	- "annealer_gpu_SI -a J_Matrix_40x40.txt -x 6.4 -y 0.001 -n 35 -m 9000 -d"

	- here -a opt for passing Jmatrix filename
	- -x starting temperature and -y is the final temperature
	- -n opt for number of temperatures in the provided range
	- -m opt for number iteration at a given temperature
	- -d opt for generating files contains final spin configuration and avg_magnetism with total cut value.



	- for example if ur J_matrix is abc_XYZ.txt use the cmdline option "-a ../bin_SI/abc_XYZ.txt -d" then the debug files are generated in same folder with name ending with _XYZ.txt

	- Another example "annealer_gpu_SI -a ../bin_SI/J_Matrix_40x40.txt -x 6.4 -y 0.001 -n 35 -m 9000 -d"