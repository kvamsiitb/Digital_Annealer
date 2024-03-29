#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <unistd.h>
#include <getopt.h>

#include <vector>
#include <chrono>

#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <time.h>
#include <algorithm>

#define THREADS 1024 //or more threads gpu crashes
#define BREAK_UPDATE_VAL 2//1000 
#define TCRIT 2.26918531421f

#include "annealer_gpu_SI/utils.hpp"

#define PERCENTAGE_CHANGE_MAX_ENERGY 0.01f
#define BREAK_AFTER_ITERATION 0.05f
//__constant__ float kd_floats[1000000];
void printVecOfVec(std::vector<float> adjMat)
{
	std::cout << "\n";
	for (int j = 0; j < sqrt(adjMat.size()); j++) {
		for (int i = 0; i < sqrt(adjMat.size()); i++)
		{
			std::cout << adjMat[i + sqrt(adjMat.size())*j] << '\t';
		}
		std::cout << "\n";
	}

}

// float atomicMin
__device__ __forceinline__ float mAtomicMin(float *address, float val)
{
	int ret = __float_as_int(*address);
	while (val < __int_as_float(ret))
	{
		int old = ret;
		if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
			break;
	}
	return __int_as_float(ret);
}

//
__device__ __forceinline__ float mAtomicMax(float *address, float val)
{
	int ret = __float_as_int(*address);
	while (val > __int_as_float(ret))
	{
		int old = ret;
		if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
			break;
	}
	return __int_as_float(ret);
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void d_debug_kernel(float* gpuAdjMat, unsigned int* gpuAdjMatSize, signed char* gpu_spins, signed char* gpu_spins_1, const unsigned int* gpu_num_spins);


__global__ void init_best_energy(float* total_energy, float* best_energy, bool init = false)
{
	if (init)
	{
		best_energy[0] = total_energy[0];
		printf("initial energy %.6f \n", total_energy[0]);
	}
	else
	{
		mAtomicMin(best_energy, total_energy[0]);
		printf(" best_energy, total_energy %.6f %.6f \n", best_energy[0], total_energy[0]);
	}
}



// Initialize lattice spins
__global__ void init_spins_total_energy(float* gpuAdjMat, unsigned int* gpuAdjMatSize,
	float* gpuLinTermsVect,
	const float* __restrict__ randvals,
	signed char* gpuSpins,
	const unsigned int* gpu_num_spins,
	float* total_energy,
	curandState * state,
	unsigned long seed);

// fINAL lattice spins
__global__ void final_spins_total_energy(float* gpuAdjMat, unsigned int* gpuAdjMatSize,
       float* gpuLinTermsVect,
       signed char* gpuSpins,
       const unsigned int* gpu_num_spins,
       float* total_energy);

__global__ void changeInLocalEnePerSpin(float* gpuAdjMat, unsigned int* gpuAdjMatSize,
	float* gpuLinTermsVect,
	const float* __restrict__ randvals,
	signed char* gpuLatSpin,
	const unsigned int* gpu_num_spins,
	const float beta,
	float* total_energy,
	curandState* globalState);
  
__global__ void d_avg_magnetism(signed char* gpuSpins, const unsigned int* gpu_num_spins, float* avg_magnetism);

// Initialize lattice spins
__global__ void preprocess_max_cut_from_ising(float* gpuAdjMat, unsigned int* gpuAdjMatSize,
	signed char* gpuSpins,
	const unsigned int* gpu_num_spins,
	float* max_cut_value,
	int* plus_one_spin,
	int* minus_one_spin);

std::vector<double> create_beta_schedule_linear(uint32_t num_sweeps, double beta_start, double beta_end = -1.f);

__device__ volatile int sem = 0;

__global__ void initSemaphore() {
	sem = 0;
}

__device__ void acquire_semaphore(volatile int *lock){
  while (atomicCAS((int *)lock, 0, 1) != 0);
}

__device__ void release_semaphore(volatile int *lock){
  *lock = 0;
  __threadfence();
}
  
static void usage(const char *pname) {

	const char *bname = nullptr;//@R = rindex(pname, '/');

	fprintf(stdout,
		"Usage: %s [options]\n"
		"options:\n"
		"\t-i|--J_Matrix_file <FILENAME>\n"
		"\t\tConnectivity matrix (no multiple connection between same nodes)\n"
		"\n"
		"\t-x|--start temperature <FLOAT>\n"
		"\t\t \n"
		"\n"
		"\t-y|--stop temperature <FLOAT>\n"
		"\t\tnumber of lattice columns\n"
		"\n"
		"\t-n|--niters <INT>\n"
		"\t\tnumber of iterations\n"
		"\n"
		"\t-n|--sweeps_per_beta <INT>\n"
		"\t\tnumber of sweep per temperature\n"
		"\n"
		"\t-s|--seed <SEED>\n"
		"\t\tfix the starting point\n"
		"\n"
		"\t-s|--debug \n"
		"\t\t Print the final lattice value and shows avg magnetization at every temperature\n"
		"\n"
		"\t-o|--write-lattice\n"
		"\t\twrite final lattice configuration to file\n\n",
		bname);
	exit(EXIT_SUCCESS);
}

int main(int argc, char* argv[])
{

  std::string filename = "";//argv[1]
  std::string linear_file = "";

  float start_temp = 20.f;
  float stop_temp = 0.001f;
  unsigned long long seed = ((getpid()* rand()) & 0x7FFFFFFFF); //((GetCurrentProcessId()* rand()) & 0x7FFFFFFFF);
  
  unsigned int num_temps = 1000; //atoi(argv[2]);
  unsigned int num_sweeps_per_beta = 1;//atoi(argv[3]);
	
 
  bool write = false;
  bool debug = false;
  

  std::cout << "Start parsing the file " << std::endl;

  while (1) {
		static struct option long_options[] = {
			{     "J_Matrix_file", required_argument, 0, 'a'},
			{ "Linear_file", required_argument, 0, 'l' },
			{     "start_temp", required_argument, 0, 'x'},
			{     "stop_temp", required_argument, 0, 'y'},
			{          "seed", required_argument, 0, 's'},
			{        "niters", required_argument, 0, 'n'},
			{ "sweeps_per_beta", required_argument, 0, 'm'},
			{ "write-lattice",       no_argument, 0, 'o'},
			{          "debug",       no_argument, 0, 'd'},
			{          "help",       no_argument, 0, 'h'},
			{               0,                 0, 0,   0}
		};

		int option_index = 0;
		int ch = getopt_long(argc, argv, "a:l:x:y:s:n:m:odh", long_options, &option_index);
		if (ch == -1) break;

		switch (ch) {
		case 0:
			break;
		case 'a':
			filename = (optarg); break;
		case 'l':
			linear_file = (optarg); break;
		case 'x':
			start_temp = atof(optarg); break;
		case 'y':
			stop_temp = atof(optarg); break;
		case 's':
			seed = atoll(optarg);
			break;
		case 'n':
			num_temps = atoi(optarg); break;
		case 'm':
			num_sweeps_per_beta = atoi(optarg); break;
		case 'o':
			write = true; break;
 		case 'd':
			debug = true; break;
		case 'h':
			usage(argv[0]); break;
		case '?':
			exit(EXIT_FAILURE);
		default:
			fprintf(stderr, "unknown option: %c\n", ch);
			exit(EXIT_FAILURE);
		}
	}

    std::cout << "filename " << filename << " linear filename " << linear_file << " start temp " << start_temp << " stop temp " << stop_temp << " seed " << seed << " num temp " << num_temps << " num sweeps " <<  num_sweeps_per_beta << std::endl;
	std::vector<float> adjMat;// float
 	double starttime = rtclock();
	ParseData parseData(filename, adjMat);

	std::vector<float> linearTermsVect;
	//if (linear_file.empty() == false)
	parseData.readLinearValues(linear_file, linearTermsVect);

	double endtime = rtclock();
  
    if(debug)
  	  printtime("ParseData time: ", starttime, endtime);

	unsigned int adj_mat_size = adjMat.size();
	auto graphs_data = parseData.getDataDims();//sqrt(adjMat.size());
	unsigned int num_spins = graphs_data.at(0);
	unsigned int CPU_THREADS = THREADS;//(num_spins < 32) ? num_spins : 32; 

//	cudaMemcpyToSymbol( &THREADS, &CPU_THREADS, sizeof(unsigned int));
	// Setup cuRAND generator
	
    std::cout << "adj_mat_size: " << adj_mat_size << " num_spins: " << num_spins << " num of temperature "<< num_temps << " num_sweeps per beta "<< num_sweeps_per_beta << std::endl;
	curandGenerator_t rng;
	
	curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	curandSetPseudoRandomGeneratorSeed(rng, seed);
	float *gpu_randvals;// same as spins
	gpuErrchk(cudaMalloc((void**)&gpu_randvals, (num_spins) * sizeof(float)));

	float *gpuLinTermsVect;
	gpuErrchk(cudaMalloc((void**)&gpuLinTermsVect, (num_spins) * sizeof(float)));

	if (linearTermsVect.size() != num_spins)
		std::cout << "	[ERROR] error in parsing the linear terms from file" << std::endl;

	gpuErrchk(cudaMemcpy(gpuLinTermsVect, linearTermsVect.data(), (num_spins) * sizeof(float), cudaMemcpyHostToDevice));

	float *gpuAdjMat;
	gpuErrchk(cudaMalloc((void**)&gpuAdjMat, (adj_mat_size) * sizeof(float)));
 
	starttime = rtclock();
	gpuErrchk(cudaMemcpy(gpuAdjMat, adjMat.data(), (adj_mat_size) * sizeof(float), cudaMemcpyHostToDevice));
  endtime = rtclock();
   
  if(debug)
     printtime("J Matrix data transfer time: ", starttime, endtime);
	//printVecOfVec(adjMat);

	unsigned int* gpu_adj_mat_size;
	gpuErrchk(cudaMalloc((void**)&gpu_adj_mat_size, sizeof(*gpu_adj_mat_size)));
	gpuErrchk(cudaMemcpy(gpu_adj_mat_size, &adj_mat_size, sizeof(*gpu_adj_mat_size), cudaMemcpyHostToDevice));

	unsigned int* gpu_num_spins;
	gpuErrchk(cudaMalloc((void**)&gpu_num_spins, sizeof(*gpu_num_spins)));
	gpuErrchk(cudaMemcpy(gpu_num_spins, &num_spins, sizeof(*gpu_num_spins), cudaMemcpyHostToDevice));
	adjMat.clear();// deallcoate vector //@ERROR

	int* gpu_plus_one_spin;
	cudaHostAlloc(&gpu_plus_one_spin, sizeof(int), 0);

	int* gpu_minus_one_spin;
	cudaHostAlloc(&gpu_minus_one_spin, sizeof(int), 0);

	int* gpu_best_plus_one_spin;
	cudaHostAlloc(&gpu_best_plus_one_spin, sizeof(int), 0);
	gpu_best_plus_one_spin[0] = 0;

	int* gpu_best_minus_one_spin;
	cudaHostAlloc(&gpu_best_minus_one_spin, sizeof(int), 0);
	gpu_best_minus_one_spin[0] = 0;

	
	float* gpu_total_energy;
	cudaHostAlloc(&gpu_total_energy, sizeof(float), 0);

	float* gpu_best_energy;
	cudaHostAlloc(&gpu_best_energy, sizeof(float), 0);

	float* gpu_max_cut_value;
	cudaHostAlloc(&gpu_max_cut_value, sizeof(float), 0);

	float* gpu_best_max_cut_value;
	cudaHostAlloc(&gpu_best_max_cut_value, sizeof(float), 0);
	gpu_best_max_cut_value[0] = -1000.f;

	float* gpu_avg_magnetism;	
	cudaHostAlloc(&gpu_avg_magnetism, sizeof(*gpu_avg_magnetism), 0);	
	gpu_avg_magnetism[0] = 0.f;
 
	// Setup spin values
	signed char *gpu_spins;
	gpuErrchk(cudaMalloc((void**)&gpu_spins, num_spins * sizeof(*gpu_spins)));

	std::cout << "initialize spin values " << std::endl;
	int blocks = (num_spins + THREADS - 1) / THREADS;
	curandGenerateUniform(rng, gpu_randvals, num_spins);
	
  //d_debug_kernel<<< 1, 1>>>(gpuAdjMat, gpu_adj_mat_size, gpu_num_spins);

// is a seed for random number generator
	time_t t;
	time(&t);
 
	//  create random states    
	curandState* devRanStates;
	cudaMalloc(&devRanStates, num_spins * sizeof(curandState));
 	
   starttime = rtclock();

	init_spins_total_energy << < num_spins, THREADS >> > (gpuAdjMat, gpu_adj_mat_size,
		gpuLinTermsVect,
		gpu_randvals,
		gpu_spins,
		gpu_num_spins,
		gpu_total_energy,
		devRanStates,
		(unsigned long)t);
  
  cudaDeviceSynchronize();
      
 	 endtime = rtclock();

	printtime("init_spins values and calculate total Energy time: ", starttime, endtime);
 

	gpuErrchk(cudaPeekAtLastError());

	gpu_best_energy[0] = gpu_total_energy[0];

	std::cout << "start annealing with initial energy: " << gpu_best_energy[0] << std::endl;
	std::vector<double> beta_schedule = create_beta_schedule_linear(num_temps, start_temp, stop_temp);


  std::string out_filename = "avgmagnet_";  
  std::string in_adjmat = filename;
  {
    // Find position of '_' using find()
    int pos = in_adjmat.find_last_of("_");
    // Copy substring after pos
    std::string sub = in_adjmat.substr(pos + 1);
    out_filename += sub;
  }

 	FILE* fptr = fopen(out_filename.c_str() , "w");

	auto t0 = std::chrono::high_resolution_clock::now();
 
// temperature 
	for (int i = 0; i < beta_schedule.size(); i++)
	{
	 int no_update = 0;
	 cudaEvent_t start, stop;
   if(debug)
   {   
     // @ Debugging
     
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
    }         
         
      for(int ii = 0; ii < num_sweeps_per_beta; ii++)
	    {   
        //int prev_energy = gpu_total_energy[0];
        initSemaphore<<<1, 1>>>();
        curandGenerateUniform(rng, gpu_randvals, num_spins);
   if(debug)
   {         
        cudaEventRecord(start); 
   }
      	changeInLocalEnePerSpin << < num_spins, THREADS >> > (gpuAdjMat, gpu_adj_mat_size,
				gpuLinTermsVect,
      			gpu_randvals,
      			gpu_spins,
      			gpu_num_spins,
      			beta_schedule.at(i),
      			gpu_total_energy,
      			devRanStates);
                        
    if(debug)
    {
         cudaEventRecord(stop);   
         cudaEventSynchronize(stop);
         float milliseconds = 0;
         cudaEventElapsedTime(&milliseconds, start, stop);
         printf("Elapse time : %f ms \n", milliseconds);
    }     
       cudaDeviceSynchronize();
       
       if(gpu_total_energy[0] > gpu_best_energy[0])
           no_update = 0;
       
       gpu_best_energy[0] = std::min(gpu_total_energy[0], gpu_best_energy[0]);
    	 
       if (  (gpu_best_energy[0] - gpu_total_energy[0]) < (PERCENTAGE_CHANGE_MAX_ENERGY) * gpu_best_energy[0])
  		  	no_update = 0;
  		 else
  		  	no_update++;
  	//	printf("cur engy %.1f best engy %.1f \n", gpu_total_energy[0], gpu_best_energy[0]);
  		if (no_update > (BREAK_AFTER_ITERATION) * num_sweeps_per_beta)
  			{
        break;
        }

              
// @R Debugging
if(debug)
{
   {	
       d_avg_magnetism << < 1, THREADS >> >(gpu_spins, gpu_num_spins, gpu_avg_magnetism);   	
   }
}     	
         cudaDeviceSynchronize();      

		 gpuErrchk(cudaPeekAtLastError());         		 
 	  }
          
  if(debug)
    fprintf(fptr, "Temperature %.6f magnet %.6f \n", 1.f/beta_schedule.at(i),  gpu_avg_magnetism[0]); 


	}
 
	auto t1 = std::chrono::high_resolution_clock::now();

	double duration = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

  fprintf(fptr, "duration %.3f \n", (duration * 1e-6) );
  fclose(fptr);

 
 // @R Debugging 
/*	d_debug_kernel << < 1, 1 >> > (gpuAdjMat,
		gpu_adj_mat_size,
		gpu_spins,
		gpu_spins_1,
		gpu_num_spins);
*/   


  
  signed char cpu_spins[num_spins];

	gpu_max_cut_value[0] = 0.f;
	gpu_plus_one_spin[0] = 0;
	gpu_minus_one_spin[0] = 0;
  gpu_total_energy[0] = 0.f;

   {

       final_spins_total_energy << < num_spins, THREADS >> > (gpuAdjMat, gpu_adj_mat_size,
                        gpuLinTermsVect,
                        gpu_spins,
                        gpu_num_spins,
                        gpu_total_energy);

       preprocess_max_cut_from_ising << < num_spins, THREADS >> > (gpuAdjMat,
  				gpu_adj_mat_size,
  				gpu_spins,
  				gpu_num_spins,
  				gpu_max_cut_value,
  				gpu_plus_one_spin,
  				gpu_minus_one_spin);
  
  			cudaDeviceSynchronize();
       gpuErrchk(cudaMemcpy(cpu_spins, gpu_spins, num_spins * sizeof(*gpu_spins), cudaMemcpyDeviceToHost));
       gpu_max_cut_value[0] *= -0.5f; 
   }     
        
			gpu_best_max_cut_value[0] = std::max(gpu_best_max_cut_value[0], gpu_max_cut_value[0]);
			gpu_best_plus_one_spin[0] = std::max(gpu_best_plus_one_spin[0], gpu_plus_one_spin[0]);
			gpu_best_minus_one_spin[0] = std::max(gpu_best_minus_one_spin[0], gpu_minus_one_spin[0]);
			printf("cur engy %.1f curr cut %.1f best cut %.1f with best +1 %d and -1 %d \n", gpu_total_energy[0], gpu_max_cut_value[0], gpu_best_max_cut_value[0], gpu_best_plus_one_spin[0], gpu_best_minus_one_spin[0]);

 if(debug)
 {
 

 
  std::string spins_filename = "spins_";  
  
  std::string adjmat = filename;

  {
    // Find position of '_' using find()
    int pos = adjmat.find_last_of("_");
    // Copy substring after pos
    std::string sub = adjmat.substr(pos + 1);
    spins_filename += sub;
  }

 	FILE* fptr1 = fopen(spins_filename.c_str() , "w");
  for(int i = 0; i < num_spins; i++)
  {
        fprintf(fptr1, "%d\t",  (int)cpu_spins[i]);

  }  
  fprintf(fptr1,"\n\n\n");
  //fprintf(fptr1,"\tbest energy value: %.6f\n", gpu_best_energy[0] );
  fprintf(fptr1,"\ttotal energy value: %.6f\n", gpu_total_energy[0] );
  fprintf(fptr1,"\tbest max cut value: %.6f\n", gpu_best_max_cut_value[0]);
	// fprintf(fptr1," \telapsed time in sec: %.6f\n", duration * 1e-6 );
  fclose(fptr1);
  
 }
	std::cout << "\ttotal energy value: " << gpu_total_energy[0] << std::endl;
	std::cout << "\tbest max cut value: " << gpu_best_max_cut_value[0] << std::endl;
	// std::cout << "\telapsed time in sec: " << duration * 1e-6 << std::endl;
 
	cudaFree(gpu_randvals);
	cudaFree(gpuAdjMat);
	cudaFree(gpu_adj_mat_size);
	cudaFree(gpu_num_spins);
	cudaFree(gpu_spins);
	return 0;
}




#define CORRECT 1

#if CORRECT

__global__ void changeInLocalEnePerSpin(float* gpuAdjMat, unsigned int* gpuAdjMatSize,
	float* gpuLinTermsVect,
	const float* __restrict__ randvals,
	signed char* gpuLatSpin,
	const unsigned int* gpu_num_spins,
	const float beta,
	float* total_energy,
	curandState* globalState) {

	unsigned int vertice_Id = blockIdx.x;
	unsigned int p_Id = threadIdx.x;    //32 worker threads 
	// for each neighour of vertex id pull the gpucurrentupdate[i] and place it in the shared memory

  __syncthreads();
  if (threadIdx.x == 0)
     acquire_semaphore(&sem);
  __syncthreads();
 
	// shared  spin_v0|spin_v1|.......|J_spin0| J_spin1| J_spin2|..
	__shared__ float sh_mem_spins_Energy[THREADS];
    sh_mem_spins_Energy[p_Id] = 0;
    __syncthreads();

	float current_spin_shared_mem;


	current_spin_shared_mem = (float)gpuLatSpin[vertice_Id];


	unsigned int stride_jump_each_vertice = sqrt((float)gpuAdjMatSize[0]);
	unsigned int num_spins = gpu_num_spins[0];
	int num_iter = int((num_spins) / THREADS) + 1;// @R (num_spins + THREADS - 1) / THREADS;

	// placing all the spins data into the shared memory..
	// hence, decouple the spins to the global spins
	for (int i = 0; i < num_iter; i++)
	{
		if (p_Id + i * THREADS < num_spins)
		{
			float current_spin;
			current_spin = (float)gpuLatSpin[p_Id + i * THREADS];
        
			sh_mem_spins_Energy[p_Id] += gpuAdjMat[p_Id + (i * THREADS) + (vertice_Id * stride_jump_each_vertice)] * (current_spin);
 		       
		}
	}
	__syncthreads();


  for (int off = blockDim.x/2; off; off /= 2) {
     if (threadIdx.x < off) {
         sh_mem_spins_Energy[threadIdx.x] += sh_mem_spins_Energy[threadIdx.x + off];
       }
   __syncthreads();
   }
   
  __syncthreads();
	
	if (p_Id == 0)
	{
    
      float local_ham_per_spin =  - 2.f * ( (-1.f * sh_mem_spins_Energy[0]) - gpuLinTermsVect[vertice_Id] ) * current_spin_shared_mem; //  final energy - current energy
	
	  float prob_ratio = exp(-1.f * beta * (local_ham_per_spin)); // exp(- (E_f - E_i) / T)
        
	  float acceptance_probability = min((float)1.f, prob_ratio);

	  if (randvals[vertice_Id] < acceptance_probability)
      {
            gpuLatSpin[vertice_Id] = (signed char)(-1.f * current_spin_shared_mem); 
      }
	}
 
 __threadfence(); 
 __syncthreads();
 if (threadIdx.x == 0)
   release_semaphore(&sem);
 __syncthreads();

}


#endif

// Initialize lattice spins
__global__ void init_spins_total_energy(float* gpuAdjMat, unsigned int* gpuAdjMatSize,
	float* gpuLinTermsVect,
	const float* __restrict__ randvals,
	signed char* gpuSpins,
	const unsigned int* gpu_num_spins,
	float* total_energy,
	curandState * state,
	unsigned long seed) {

	unsigned int vertice_Id = blockIdx.x; // actual spin id in this threadBlock
	unsigned int p_Id = threadIdx.x;// which worker id

	if (p_Id == 0)
	{
		float randval = randvals[vertice_Id];
		signed char val = (randval < 0.5f) ? -1 : 1;
		gpuSpins[vertice_Id] = val;// random spin init.
		curand_init(seed, blockIdx.x, 0, &state[blockIdx.x]);
	}
	__syncthreads();

	__shared__ float sh_mem_spins_Energy[THREADS];
    sh_mem_spins_Energy[p_Id] = 0;
    __syncthreads();
  
	unsigned int stride_jump_each_vertice = sqrt((float)gpuAdjMatSize[0]);
	unsigned int num_spins = gpu_num_spins[0];
	int num_iter = (num_spins + THREADS - 1) / THREADS;

	// num_iter data chucks 
	for (int i = 0; i < num_iter; i++)
	{
		// p_Id (worker group)
		if (p_Id + i * THREADS < num_spins)
		{
			sh_mem_spins_Energy[p_Id] += (- 0.5f ) * gpuAdjMat[p_Id + (i * THREADS) + (vertice_Id * stride_jump_each_vertice)] * ((float)gpuSpins[p_Id + i * THREADS]); 
		}
	}
	__syncthreads();


  for (int off = blockDim.x/2; off; off /= 2) {
     if (threadIdx.x < off) {
         sh_mem_spins_Energy[threadIdx.x] += sh_mem_spins_Energy[threadIdx.x + off];
       }
   __syncthreads();
   }
 

	if (p_Id == 0)
	{
 
		float vertice_energy = ((float)gpuSpins[vertice_Id]) * ( sh_mem_spins_Energy[0] - gpuLinTermsVect[vertice_Id] );
		// hamiltonian_per_spin[vertice_Id] = vertice_energy;// each threadblock updates its own memory location

//		printf("vertice_energy  %f \n", vertice_energy);
		atomicAdd(total_energy, vertice_energy);
	}

	//        printf("%d total %.1f",blockIdx.x, total_energy);
}

// fINAL lattice spins
__global__ void final_spins_total_energy(float* gpuAdjMat, unsigned int* gpuAdjMatSize,
	float* gpuLinTermsVect,
	signed char* gpuSpins,
	const unsigned int* gpu_num_spins,
	float* total_energy) {

	unsigned int vertice_Id = blockIdx.x; // actual spin id in this threadBlock
	unsigned int p_Id = threadIdx.x;// which worker id

	__shared__ float sh_mem_spins_Energy[THREADS];
	sh_mem_spins_Energy[p_Id] = 0;
	__syncthreads();

	unsigned int stride_jump_each_vertice = sqrt((float)gpuAdjMatSize[0]);
	unsigned int num_spins = gpu_num_spins[0];
	int num_iter = (num_spins + THREADS - 1) / THREADS;

	// num_iter data chucks 
	for (int i = 0; i < num_iter; i++)
	{
		// p_Id (worker group)
		if (p_Id + i * THREADS < num_spins)
		{
			sh_mem_spins_Energy[p_Id] += (-0.5f) * gpuAdjMat[p_Id + (i * THREADS) + (vertice_Id * stride_jump_each_vertice)] * ((float)gpuSpins[p_Id + i * THREADS]);
		}
	}
	__syncthreads();


	for (int off = blockDim.x / 2; off; off /= 2) {
		if (threadIdx.x < off) {
			sh_mem_spins_Energy[threadIdx.x] += sh_mem_spins_Energy[threadIdx.x + off];
		}
		__syncthreads();
	}


	if (p_Id == 0)
	{

		float vertice_energy = ((float)gpuSpins[vertice_Id]) * ( sh_mem_spins_Energy[0] - gpuLinTermsVect[vertice_Id] );
		// hamiltonian_per_spin[vertice_Id] = vertice_energy;// each threadblock updates its own memory location

		//printf("vertice_energy  %d %f \n",vertice_Id, vertice_energy);
		atomicAdd(total_energy, vertice_energy);
	}

	//        printf("%d total %.1f",blockIdx.x, total_energy);
}

// Initialize lattice spins
__global__ void preprocess_max_cut_from_ising(float* gpuAdjMat, unsigned int* gpuAdjMatSize,
	signed char* gpuSpins,
	const unsigned int* gpu_num_spins,
	float* max_cut_value,
	int* plus_one_spin,
	int* minus_one_spin) {

	unsigned int vertice_Id = blockIdx.x; // actual spin id in this threadBlock
	unsigned int p_Id = threadIdx.x;// which worker id
	float current_spin_row = (float)gpuSpins[vertice_Id];

	__shared__ float sh_mem_spins_Energy[THREADS];
    sh_mem_spins_Energy[p_Id] = 0;
    __syncthreads();

	unsigned int stride_jump_each_vertice = sqrt((float)gpuAdjMatSize[0]);
	unsigned int num_spins = gpu_num_spins[0];
	int num_iter = (num_spins + THREADS - 1) / THREADS;

	// num_iter data chucks 
	for (int i = 0; i < num_iter; i++)
	{
		// p_Id (worker group)
		if (p_Id + i * THREADS < num_spins)
		{
			sh_mem_spins_Energy[p_Id] += gpuAdjMat[p_Id + (i * THREADS) + (vertice_Id * stride_jump_each_vertice)] * (1.f - (current_spin_row * (float)gpuSpins[p_Id + i * THREADS]));
		}
	}
	__syncthreads();

  for (int off = blockDim.x/2; off; off /= 2) {
     if (threadIdx.x < off) {
         sh_mem_spins_Energy[threadIdx.x] += sh_mem_spins_Energy[threadIdx.x + off];
       }
   __syncthreads();
   }
   
	if (p_Id == 0)
	{

		float vertice_energy;

		vertice_energy = (0.5f) * sh_mem_spins_Energy[0];

		atomicAdd(max_cut_value, vertice_energy);

		if (current_spin_row == 1.f)
			atomicAdd(plus_one_spin, 1);
		else
			atomicAdd(minus_one_spin, 1);
	}

	//       
}


std::vector<double> create_beta_schedule_linear(uint32_t num_sweeps, double beta_start, double beta_end)
{
	std::vector<double> beta_schedule;
	double beta_max;
	if (beta_end == -1)
		beta_max = (1/1000)*beta_start;//  here temperature will be zero when beta_max is 1000.f
	else
		beta_max = beta_end;
	double diff = (beta_start - beta_max) / (num_sweeps - 1);// A.P 3.28 - 0.01 inverse value increa finnal decrease
	for (int i = 0; i < num_sweeps; i++)
	{
		double val = beta_start - (i)*diff;
		beta_schedule.push_back(( 1.f /val));
	}
	
	return beta_schedule;
}

__global__ void d_debug_kernel(float* gpuAdjMat, unsigned int* gpuAdjMatSize, signed char* gpu_spins, signed char* gpu_spins_1, const unsigned int* gpu_num_spins)
{
	int ones = 0;
	int ones_1 = 0;
	for (int i = 0; i < gpu_num_spins[0]; i++)
	{
		printf("%d %.1f ", i, (float)gpu_spins[i]);
		if ((float)gpu_spins[i] == 1.f)
			ones++;
		if ((float)gpu_spins_1[i] == -1.f)
			ones_1++;
	}

	printf("\n");
	printf("\n");
	printf("%d %d \n", ones, ones_1);
	int m_ones = 0;
	int m_ones_1 = 0;
	for (int i = 0; i < gpu_num_spins[0]; i++)
	{
		printf("%d %.1f ", i, (float)gpu_spins_1[i]);
		if ((float)gpu_spins[i] == 1.f)
			m_ones++;
		if ((float)gpu_spins_1[i] == -1.f)
			m_ones_1++;
	}
	printf("\n");
	printf("\n");
	printf("%d %d\n", m_ones, m_ones_1);
}

__global__ void d_avg_magnetism(signed char* gpuSpins, const unsigned int* gpu_num_spins, float* avg_magnetism)	
{	
  unsigned int p_Id = threadIdx.x;	
  	
	__shared__ float sh_mem_spins_Energy[THREADS];	
  sh_mem_spins_Energy[p_Id] = 0;	
  __syncthreads();	
  	
 	int num_iter = (gpu_num_spins[0] + THREADS - 1) / THREADS;
   	 	
	for (int i = 0; i < gpu_num_spins[0]; i++)	
	{	
		// p_Id (worker group)	
		if (p_Id + i * THREADS < gpu_num_spins[0])	
		{		
			sh_mem_spins_Energy[p_Id] += ((float)gpuSpins[p_Id + i * THREADS]); 	
		}	
	}	
	__syncthreads();	
 	
   for (int off = blockDim.x/2; off; off /= 2) {	
     if (threadIdx.x < off) {	
         sh_mem_spins_Energy[threadIdx.x] += sh_mem_spins_Energy[threadIdx.x + off];	
       }	
   __syncthreads();	
   }	
   	
	if (p_Id == 0)	
	{	
      avg_magnetism[0] = sh_mem_spins_Energy[0]/gpu_num_spins[0];		
  }	
}