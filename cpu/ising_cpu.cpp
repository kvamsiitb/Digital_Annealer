#include "utils.hpp"

#include <algorithm>
#include <cassert>

/* 0:Serial, 1:Parallel */
#define PARALLEL 0

/* 0:Standard, 1:Algorithm1, 2:Algorithm2, 3:Metropolis-Hastings */
#define IMPLEMENTATION 3

/* 0:InOrder, 1:Random */
#define RANDOM_SELECTION_SPINS 1

int main(int argc, char **argv)
{
    bool debug = true;

    int num_sweeps_per_beta = 1000;
    int num_temps = 35;
    float startTemp = 2.9f;

    if (argc != 6)
    {
        // ./ising_cpu num_sweeps_per_beta startTemp num_temps adj_mat linear_mat    
        printf("error in the code");
        exit(-1);
    }

    startTemp = atof(argv[1]);
    num_temps = atoi(argv[2]);
    num_sweeps_per_beta = atoi(argv[3]);
    std::string filename = argv[4];
    std::string linear_file = argv[5];

    double starttime = rtclock();

    std::vector<float> adjMat;
    ParseData parseData(filename, adjMat);

    std::vector<float> linearTermsVect;
    if (linear_file.empty() == false)
        parseData.readLinearValues(linear_file, linearTermsVect);

    double endtime = rtclock();

    unsigned int adj_mat_size = adjMat.size();
    auto graphs_data = parseData.getDataDims(); // sqrt(adjMat.size());
    unsigned int num_spins = graphs_data.at(0);

    assert(float(adj_mat_size) == std::pow(float(num_spins), 2.0));
    assert(num_spins == (unsigned int)linearTermsVect.size());
    if (debug)
        printtime("ParseData time: ", starttime, endtime);

    // std::random_device dev;
    // std::mt19937 rng(dev()); // defined by header file itself instead
    std::uniform_int_distribution<> int_dist(0, num_spins - 1);

    std::vector<float> avg_magnet;
    //printVecOfVec(adjMat);
    //printf("\n\n\n");
    //printVecOfVec(linearTermsVect);
    std::vector<signed char> spinVec;
    std::vector<float> localEnergyPerSpin;
    spinVec.resize(num_spins);
    localEnergyPerSpin.resize(num_spins);

    initializeSpinVec(spinVec);
    //debugSpinVal(spinVec);

    std::cout << "\nStarting annealing with startTemp: " << startTemp << ", num_temps: " << num_temps
        << ", num_sweeps_per_beta: " << num_sweeps_per_beta << ", files: " << filename << " " << linear_file
        << "\n" << "Threading: " << (PARALLEL ? "Parallel" : "Serial")
        << ", Random Spin Selection: " << (RANDOM_SELECTION_SPINS ? "On" : "Off") 
        << ", Implementation: " << IMPLEMENTATION << "\n" << std::endl;

    std::vector<double> beta_schedule = create_beta_schedule_linear(num_temps, startTemp, 0.001f);
    std::vector<double> fp_schedule = create_fp_schedule_linear(num_temps);

    std::string out_filename = "avgmagnet_";
    std::string in_adjmat = filename;

    // Find position of '_' using find()
    int pos = in_adjmat.find_last_of("_");
    // Copy substring after pos
    std::string sub = in_adjmat.substr(pos + 1);
    out_filename += sub;

    FILE *fptr = fopen(out_filename.c_str(), "w");

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<int> random_indices(spinVec.size());
    std::iota(random_indices.begin(), random_indices.end(), 0);

    std::default_random_engine re{ rd() };
    unsigned int current_spinIdx = 0;

    for (int i = 0; i < beta_schedule.size(); i++)
    {
        for (int j = 0; j < num_sweeps_per_beta; j++)
        {
#if PARALLEL
            std::vector<unsigned int> spinList(spinVec.size());
            
            for (int spinIdx = 0; spinIdx < spinVec.size(); spinIdx++)
            {
                spinList[spinIdx] = RANDOM_SELECTION_SPINS ? int_dist(rng) : spinIdx;
                changeInLocalEnePerSpin(adjMat, linearTermsVect, spinVec, localEnergyPerSpin, spinList[spinIdx]);
            }

            for (int spinIdx = 0; spinIdx < spinVec.size(); spinIdx++)
            {
                updateMetropolisHasting(spinVec, localEnergyPerSpin, spinList[spinIdx],
                                        beta_schedule.at(i), fp_schedule.at(i), IMPLEMENTATION);
            }
#else /* not PARALLEL = SERIAL */
            for (int spinIdx = 0; spinIdx < spinVec.size(); spinIdx++)
            {
                current_spinIdx = RANDOM_SELECTION_SPINS ? int_dist(rng) : spinIdx;

                changeInLocalEnePerSpin(adjMat, linearTermsVect, spinVec, localEnergyPerSpin, current_spinIdx);
                updateMetropolisHasting(spinVec, localEnergyPerSpin, current_spinIdx,
                                        beta_schedule.at(i), fp_schedule.at(i), IMPLEMENTATION);
            }
#endif /* PARALLEL or SERIAL */

#if IMPLEMENTATION == 1 /* Algorithm 1 */
            std::shuffle(random_indices.begin(), random_indices.end(), re);
            for (int spinIdx = 0; spinIdx < fp_schedule.at(i) * spinVec.size(); spinIdx++)
            {
                spinVec[random_indices[spinIdx]] = -spinVec[random_indices[spinIdx]];
            }
#endif /* Algorithm 1 */
            }

        float magnet = avgMagnetisation(spinVec, beta_schedule.at(i));
        if (debug)
            fprintf(fptr, "Temperature %.6f magnet %.6f \n", 1.f / beta_schedule.at(i), magnet);
        avg_magnet.push_back(magnet);
        }

    auto t1 = std::chrono::high_resolution_clock::now();

    double duration = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    fprintf(fptr, "duration %.3f \n", (duration * 1e-6));
    fclose(fptr);

    printf("\n\telapsed time   : %f sec\n", duration * 1e-6);
    printf("\tupdates per ns : %f\n", (double)(spinVec.size()) * num_sweeps_per_beta / duration * 1e-3);

    if (debug)
    {
        std::string spins_filename = "spins_";
        std::string adjmat = filename;

        // Find position of '_' using find()
        int pos = adjmat.find_last_of("_");
        // Copy substring after pos
        std::string sub = adjmat.substr(pos + 1);
        spins_filename += sub;

        FILE *fptr1 = fopen(spins_filename.c_str(), "w");
        for (unsigned int spinIdx = 0; spinIdx < spinVec.size(); spinIdx++)
        {
            fprintf(fptr1, "%d\t", (int)spinVec[spinIdx]);
        }

        fprintf(fptr1, "\n\n\n");

        for (unsigned int spinIdx = 0; spinIdx < spinVec.size(); spinIdx++)
        {
            changeInLocalEnePerSpin(adjMat, linearTermsVect, spinVec, localEnergyPerSpin, spinIdx);
        }

        float total_energy = totalEnergy(adjMat, linearTermsVect, spinVec);
        fprintf(fptr1, "\ttotal energy   : %.6f\n", total_energy);

        fclose(fptr1);
        printf("\ttotal energy   : %f\n\n", total_energy);
    }

    return 0;
    }
