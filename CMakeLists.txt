cmake_minimum_required(VERSION 3.8)
project(Annealer LANGUAGES C CXX CUDA)

foreach (lang CXX CUDA)
  set(CMAKE_${lang}_STANDARD 11)
  set(CMAKE_${lang}_STANDARD_REQUIRED ON)
  set(CMAKE_${lang}_EXTENSIONS OFF)
endforeach()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

find_package(CUDALibs REQUIRED)

add_subdirectory(compiler_info)

add_subdirectory(annealer_gpu_SI)
