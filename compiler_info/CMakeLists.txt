
#Goal for this example:
# Show how CUDALibs is a replacement for FindCUDA ability to find cuda
# libraries

message("*******Fails here" ${CUDA_TOOLKIT_ROOT_DIR})
#Show the CMake Variables that are generated for the CUDA language
message(STATUS "The CUDA compiler ID: ${CMAKE_CUDA_COMPILER_ID}")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "5")
  message(STATUS "CUDA compiler version is 4.X or lower")
elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "6")
  message(STATUS "CUDA compiler version is 5.X")
elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "7")
  message(STATUS "CUDA compiler version is 6.X")
elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "8")
  message(STATUS "CUDA compiler version is 7.X")
elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9")
  message(STATUS "CUDA compiler version is 8.X")
else()
  message(STATUS "CUDA compiler version > 8")
endif()

message("*******Not Fails here")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${Annealer_SOURCE_DIR}/cmake")
find_package(CUDALibs)

message(STATUS "CUDA_VERSION_MAJOR: ${CUDA_VERSION_MAJOR}")
message(STATUS "CUDA_VERSION_MINOR: ${CUDA_VERSION_MINOR}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
message(STATUS "CUDA_VERSION_STRING: ${CUDA_VERSION_STRING}")
message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")


message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA_cupti_LIBRARY: ${CUDA_cupti_LIBRARY}")
message(STATUS "CUDA_curand_LIBRARY: ${CUDA_curand_LIBRARY}")
message(STATUS "CUDA_cusolver_LIBRARY: ${CUDA_cusolver_LIBRARY}")
message(STATUS "CUDA_cusparse_LIBRARY: ${CUDA_cusparse_LIBRARY}")
message(STATUS "CUDA_npp_LIBRARY: ${CUDA_npp_LIBRARY}")
message(STATUS "CUDA_nppc_LIBRARY: ${CUDA_nppc_LIBRARY}")
message(STATUS "CUDA_nppi_LIBRARY: ${CUDA_nppi_LIBRARY}")
message(STATUS "CUDA_npps_LIBRARY: ${CUDA_npps_LIBRARY}")
message(STATUS "CUDA_nvcuvenc_LIBRARY: ${CUDA_nvcuvenc_LIBRARY}")
message(STATUS "CUDA_nvcuvid_LIBRARY: ${CUDA_nvcuvid_LIBRARY}")
