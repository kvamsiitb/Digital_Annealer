# Digital Annealer On Aqua
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
## Submit Job In Aqua


First copy the J_matrix text file in bin_SI folder

Open anneal.cmd file 
	- In line 25, only update this line "pw01_1000_1.txt -x 14.4 -y 0.001 -n 16000 -m 1 -d" and remaining everything remains the same.
	- for example if ur J_matrix is abc_XYZ.txt use the cmdline option "-a ../bin_SI/abc_XYZ.txt -d" then the debug files are generated in build folder
	- qsub anneal.cmd
	- for checking the status use the cmd "qstat JOB_ID"



