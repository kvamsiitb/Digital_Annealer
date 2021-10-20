/*
Class: Parse data
Desc: Parse Jij matrix
*/

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <math.h>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>

#define MH 1

using std::string;

double rtclock();

void printtime(const char *str, double starttime, double endtime);

class ParseData
{
public:
    ParseData(const string filename, std::vector<float> &adjMat);
    void readDataDim(string data, std::vector<float> &adjMat);
    void readData(string data, std::vector<float> &adjMat);
    void readLinearValues(const string filename, std::vector<float> &linearVect);

    std::vector<unsigned int> getDataDims() const;

private:
    std::unique_ptr<std::ifstream, std::function<void(std::ifstream *)>> _pifstream;
    //std::vector<int> _matA;
    std::vector<unsigned int> _data_dims; // rows and columns
};

/*
End of parser
*/

void printVecOfVec(std::vector<float> adjMat);

std::vector<double> create_beta_schedule_linear(uint32_t num_sweeps, double beta_start, double beta_end = 0.001f);

float avgMagnetisation(const std::vector<signed char> &spinVec, float temp);

void initializeSpinVec(std::vector<signed char> &spinVec);

void changeInLocalEnePerSpin(const std::vector<float> &adjMat, std::vector<float> &linearTermsVect, unsigned int adj_mat_size,
    const std::vector<signed char> &spinVec, unsigned int num_spins,
    std::vector<float> &localEnergyPerSpin,
    unsigned int spinIdx);

void updateMetropolisHasting(std::vector<signed char> &spinVec, unsigned int num_spins,
    const std::vector<float> &localEnergyPerSpin,
    unsigned int spinIdx, float beta);

float partialMaxCut(const std::vector<float> &adjMat, std::vector<float> &linearTermsVect, unsigned int adj_mat_size,
    const std::vector<signed char> &spinVec, unsigned int num_spins,
    unsigned int spinIdx);

void debugSpinVal(std::vector<signed char> &spinVec);

void updateMetropolisHasting(std::vector<signed char> &spinVec, unsigned int num_spins,
    const std::vector<float> &localEnergyPerSpin,
    unsigned int spinIdx, float beta)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dist(0, 1);

#if MH
    float prob_ratio = exp(-1.f * beta * (localEnergyPerSpin[spinIdx])); // exp(- (E_f - E_i ) / T)
    float acceptance_probability = std::min((float)1.f, prob_ratio);

    // H(new_spin) - H(old_spin) = - 2 * S * s_i

    float randval = dist(gen);
    if (randval < acceptance_probability)
    {
        spinVec[spinIdx] = (signed char)(-1 * (int)spinVec[spinIdx]);
    }
#endif /* MH */
}

void changeInLocalEnePerSpin(const std::vector<float> &adjMat, std::vector<float> &linearTermsVect, unsigned int adj_mat_size,
    const std::vector<signed char> &spinVec, unsigned int num_spins,
    std::vector<float> &localEnergyPerSpin,
    unsigned int spinIdx)
{
    float changeInEnergy = 0.f;
    for (int index = 0; index < num_spins; index++)
    {
        changeInEnergy += -1.f * adjMat[num_spins * spinIdx + index] * (float)spinVec[index]; // S = - \sum Jij[] s[j] - h[i]
    }

    changeInEnergy += -1.f * linearTermsVect[spinIdx];

    // changeInEnergy(i.e. (E_f - E_i ) ) = S * -1 * s[i] - S * s[i] = -2 * S * s[i]
    changeInEnergy = -2.f * changeInEnergy * (float)spinVec[spinIdx];
    localEnergyPerSpin[spinIdx] = changeInEnergy;
}

float avgMagnetisation(const std::vector<signed char> &spinVec, float beta)
{
    float ones = 0;
    for (int i = 0; i < spinVec.size(); i++)
    {
        //printf("%d %.1f ", i, (float)spinVec[i]);
        ones += (float)spinVec[i];
    }
    float avg_magnet = ones / spinVec.size();
    
    printf(" temp: %f\tmagnetization: % g\n", 1.f / beta, avg_magnet);

    return avg_magnet;
}

float partialMaxCut(const std::vector<float> &adjMat, std::vector<float> &linearTermsVect, unsigned int adj_mat_size,
    const std::vector<signed char> &spinVec, unsigned int num_spins,
    unsigned int spinIdx)
{
    float pMaxCut = 0.f;
    for (int index = 0; index < num_spins; index++)
    {
        pMaxCut += adjMat[num_spins * spinIdx + index] * (1.f - ((float)spinVec[index] * (float)spinVec[spinIdx]));
    }

    // changeInEnergy(i.e. (E_f - E_i ) ) = S * -1 * s[i] - S * s[i] = -2 * S * s[i]
    pMaxCut = 0.5f * pMaxCut;

    return pMaxCut;
}

void initializeSpinVec(std::vector<signed char> &spinVec)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::bernoulli_distribution d(0.5f);

    for (int i = 0; i < spinVec.size(); i++)
    {
        bool bern_value = d(gen);
        if (bern_value)
        {
            spinVec[i] = (signed char)1;
        }
        else
        {
            spinVec[i] = (signed char)-1;
        }
    }
}

std::vector<double> create_beta_schedule_linear(uint32_t num_sweeps, double beta_start, double beta_end)
{
    std::vector<double> beta_schedule;
    double beta_max;
    if (beta_end == -1)
        beta_max = (1 / 1000) * beta_start; //  here temperature will be zero when beta_max is 1000.f
    else
        beta_max = beta_end;
    double diff = (beta_start - beta_max) / (num_sweeps - 1); // A.P 3.28 - 0.01 inverse value increa finnal decrease
    for (int i = 0; i < num_sweeps; i++)
    {
        double val = beta_start - (i)*diff;
        beta_schedule.push_back((1.f / val));
    }

    return beta_schedule;
}

void printVecOfVec(std::vector<float> adjMat)
{
    std::cout << "\n";

    for (int j = 0; j < sqrt(adjMat.size()); j++)
    {
        for (int i = 0; i < sqrt(adjMat.size()); i++)
        {
            std::cout << adjMat[i + sqrt(adjMat.size()) * j] << '\t';
        }

        std::cout << "\n";
    }
}

void debugSpinVal(std::vector<signed char> &spinVec)
{
    std::cout << "\n";
    for (int j = 0; j < spinVec.size(); j++)
    {
        std::cout << (int)spinVec[j] << '\t';
    }
    std::cout << std::endl;
}

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);

    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime)
{
    printf("%s%3f seconds\n", str, endtime - starttime);
}

ParseData::ParseData(const string filename, std::vector<float> &adjMat) : _pifstream(new std::ifstream(filename, std::ifstream::in), [](std::ifstream *fp) { fp->close(); })
{
    int j = 0;
    string row_line;
    if (_pifstream->is_open())
    {
        while (std::getline(*_pifstream, row_line))
        {
            if (j == 0)
                readDataDim(row_line, adjMat);
            else
            {
                if (j <= _data_dims.at(1) && j != 0)
                    readData(row_line, adjMat);
            }
            //std::cout << "Each row "<< filename << " " << row_line << std::endl;
            j++;
        }
        _pifstream->close();
    }
    else
    {
        std::cerr << "[ERROR] Coupling File not opening" << std::endl;
    }
}

void ParseData::readLinearValues(const string filename, std::vector<float> &linearVect)
{
    std::unique_ptr<std::ifstream, std::function<void(std::ifstream *)>> pLVifstream(
        new std::ifstream(filename, std::ifstream::in),
        [](std::ifstream *fp) { fp->close(); });

    string row_line;
    if (pLVifstream->is_open())
    {
        while (std::getline(*pLVifstream, row_line))
        {
            std::istringstream input;
            input.str(row_line);

            for (std::string line; std::getline(input, line, ' ');)
            {
                linearVect.push_back(std::stof(line));
            }
        }
        pLVifstream->close();
    }
    else
    {
        std::cerr << " [ERROR] Linear File not opening" << std::endl;
    }

    // print the value of the vector
}

void ParseData::readDataDim(string data, std::vector<float> &adjMat)
{
    std::istringstream input;
    input.str(data);

    for (std::string line; std::getline(input, line, ' ');)
    {
        //std::cout << line << " *****  Dimensions Verrtices and Edges **** " << std::endl;
        _data_dims.push_back(std::stoi(line));
        //* std::cout << std::stoi(line) << "*****  Dimensions Verrtices and Edges **** " << std::endl;
        //std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    adjMat.resize(_data_dims[0] * _data_dims[0]);
}

void ParseData::readData(string data, std::vector<float> &adjMat)
{
    std::istringstream input;
    input.str(data);
    std::vector<std::string> line_data; //@R std::vector<float> line_data;
    for (std::string line; std::getline(input, line, ' ');)
    {
        //std::cout << line << std::endl;
        line_data.push_back(line);
    }
    if (line_data.size() == 3)
    {
        //std::cout << line_data.size() << std::endl;
        //std::cout << line_data.at(0) <<  " " << line_data.at(1) << " " << line_data.at(2) << std::endl;
        //std::this_thread::sleep_for(std::chrono::seconds(1));
        int first_entry = std::stoi(line_data.at(0));
        int sec_entry = std::stoi(line_data.at(1));
        adjMat[(_data_dims.at(0) * (first_entry - 1)) + (sec_entry - 1)] = stof(line_data.at(2));
        //std::cout << adjMat[_data_dims.at(0)*(line_data.at(0) - 1) + (line_data.at(1) - 1)] << std::endl;
    }
}

std::vector<unsigned int> ParseData::getDataDims() const
{
    return _data_dims;
} // Parse class ends

#if 0 // Nearest Neighbor
void initializeMat(vector<signed char> &matA, size_t size_matA)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::bernoulli_distribution d(0.5f);

    for (int i = 0; i < matA.size(); i++)
    {
        bool bern_value = d(gen);
        if (bern_value)
        {
            matA[i] = 1;
        }
        else
        {
            matA[i] = -1;
        }
    }
}
void updateMat(vector<signed char> &mat, float inv_temp, size_t shift)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> uniform(0.0f, 1.0f);

    for (int i = 0; i < mat.size(); i++)
    {

        // *_
        size_t right_jump = i + 1;
        signed char right_val = 0;
        if (right_jump < mat.size())
            right_val = mat.at(right_jump);

        // _*
        size_t left_jump = i - 1;
        signed char left_val = 0;
        if (left_jump > -1)
            left_val = mat.at(left_jump);

        // top____________*
        size_t top_jump = i - shift;
        signed char top_val = 0;
        if (top_jump > -1)
            top_val = mat.at(top_jump);

        // *________________ bottom
        size_t bottom_jump = i + shift;
        signed char bottom_val = 0;
        if (bottom_jump < mat.size())
            bottom_val = mat.at(bottom_jump);

        signed char nn_sum = right_val + left_val + top_val + bottom_val;
        signed char lij = mat.at(i);

        float acceptance_ratio = exp(-2.0f * inv_temp * nn_sum * lij);

        if (uniform(gen) < acceptance_ratio)
        {
            mat[i] = -lij;
        }

    }
}
#endif /* #if 0 */
