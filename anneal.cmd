#!/bin/bash
#PBS -e errorfile.err
#PBS -o logfile.log
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q gpuq
#PBS -M ee20s025@smail.iitm.ac.in
tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .
module purge
module avail > output.txt
nvcc --version >> output.txt
module load cmake3.18
module load gcc640
module load cuda10.1
mkdir $tempdir/build
cd $tempdir/build
cmake ..
echo $tempdir/build >> output.txt
ls  >> output.txt
make install
#../bin_SI/annealer_gpu_SI -a ../bin_SI/J_Matrix_40x40.txt -x 6.4 -y 0.001 -n 35 -m 9000 -d >> output.txt 
../bin_SI/annealer_gpu_SI -a ../bin_SI/pw01_1000_1.txt -x 14.4 -y 0.001 -n 16000 -m 1 -d >> output.txt
#mv * $PBS_O_WORKDIR/.
#rmdir $tempdir
