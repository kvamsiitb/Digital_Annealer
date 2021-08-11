#!/bin/bash

# code to build on EE server iitm

mkdir build
cd build
cmake ..
make install
cd ../bin_SI