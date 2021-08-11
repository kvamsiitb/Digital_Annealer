#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <functional>
#include <memory>

#include <sys/time.h>
using std::string;

double rtclock();

void printtime(const char *str, double starttime, double endtime);

class ParseData
{
public:
	ParseData(const string filename, std::vector<float>& adjMat);
	void readDataDim(string data, std::vector<float>& adjMat);
	void readData(string data, std::vector<float>& adjMat);
	void readLinearValues(const string filename, std::vector<float>& linearVect);

	std::vector<unsigned int> getDataDims() const;
private:
	std::unique_ptr<std::ifstream, std::function<void(std::ifstream*)> > _pifstream;
	//std::vector<int> _matA;
	std::vector<unsigned int> _data_dims;// rows and columns
};
