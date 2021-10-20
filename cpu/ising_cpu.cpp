#include "utils.hpp"

#include <cassert>

#define SERIAL 0
#define PARALLEL 1
#define RANDOM_SELECTION_SPINS 0

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
    auto graphs_data = parseData.getDataDims();//sqrt(adjMat.size());
    unsigned int num_spins = graphs_data.at(0);

    assert(float(adj_mat_size) == std::pow(float(num_spins), 2.0));
    assert(num_spins == (unsigned int)linearTermsVect.size());
    if (debug)
        printtime("ParseData time: ", starttime, endtime);

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> int_dist(0, num_spins - 1);

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
    std::cout << "\nStarting annealing with initial energy startTemp: " << startTemp << " num_temps: "
        << num_temps << " num_sweeps_per_beta: " << num_sweeps_per_beta << " files: "
        << filename << " " << linear_file << "\n" << std::endl;
    std::vector<double> beta_schedule = create_beta_schedule_linear(num_temps, startTemp, 0.001f);

    std::string out_filename = "avgmagnet_";
    std::string in_adjmat = filename;

    // Find position of '_' using find()
    int pos = in_adjmat.find_last_of("_");
    // Copy substring after pos
    std::string sub = in_adjmat.substr(pos + 1);
    out_filename += sub;

    FILE *fptr = fopen(out_filename.c_str(), "w");

    auto t0 = std::chrono::high_resolution_clock::now();

    int current_spinIdx = 0;
    for (int i = 0; i < beta_schedule.size(); i++)
    {
        for (int ii = 0; ii < num_sweeps_per_beta; ii++)
        {
#if SERIAL         
            for (int spinIdx = 0; spinIdx < spinVec.size(); spinIdx++)
            {
                current_spinIdx = RANDOM_SELECTION_SPINS ? int_dist(rng) : spinIdx;

                changeInLocalEnePerSpin(adjMat, linearTermsVect, adj_mat_size,
                    spinVec, num_spins,
                    localEnergyPerSpin,
                    current_spinIdx);
                updateMetropolisHasting(spinVec, num_spins, localEnergyPerSpin, current_spinIdx, beta_schedule.at(i));
            }
#endif /* SERIAL */

#if PARALLEL         
            for (int spinIdx = 0; spinIdx < spinVec.size(); spinIdx++)
            {
                changeInLocalEnePerSpin(adjMat, linearTermsVect, adj_mat_size,
                    spinVec, num_spins,
                    localEnergyPerSpin,
                    spinIdx);
            }

            for (int spinIdx = 0; spinIdx < spinVec.size(); spinIdx++)
            {
                current_spinIdx = RANDOM_SELECTION_SPINS ? int_dist(rng) : spinIdx;
                updateMetropolisHasting(spinVec, num_spins, localEnergyPerSpin, current_spinIdx, beta_schedule.at(i));
        }
#endif /* PARALLEL */
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

    printf("\n\telapsed time: %f sec\n", duration * 1e-6);
    printf("\tupdates per ns: %f\n", (double)(spinVec.size()) * num_sweeps_per_beta / duration * 1e-3);

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
        float total_energy = 0.f;

        for (unsigned int spinIdx = 0; spinIdx < spinVec.size(); spinIdx++)
        {
            changeInLocalEnePerSpin(adjMat, linearTermsVect, adj_mat_size,
                spinVec, num_spins,
                localEnergyPerSpin,
                spinIdx);
            total_energy += localEnergyPerSpin[spinIdx];
        }

        fprintf(fptr1, "\ttotal energy value: %.6f\n", total_energy);

        fclose(fptr1);
        printf("\ttotal energy value: %f\n\n", total_energy);
    }

    return 0;
}
