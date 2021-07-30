
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu PROPERTIES LANGUAGE CUDA)

#set(CMAKE_CUDA_FLAGS "--generate-code arch=compute_30,code=sm_70 -Xptxas=-v -c -O3 -lineinfo -Xcompiler=-Wall")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; 	-O3 	-gencode arch=compute_62,code=sm_62 )
#set(debug_compile_flags "--generate-code arch=compute_20,code=sm_20 -Xcompiler=-Werror")
if (CMAKE_BUILD_TYPE_UPPER STREQUAL "RELEASE")
    set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} -O3)
elseif (CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -g -G --generate-line-info)
endif (CMAKE_BUILD_TYPE_UPPER STREQUAL "RELEASE")

#find_package(CUDALibs REQUIRED)
message(${CUDA_curand_LIBRARY}, "&&&&&&&&&&&&&&&&")
set(TARGET "annealer_gpu_SI")

set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cc)
set(headers ${CMAKE_CURRENT_SOURCE_DIR}/include/annealer_gpu_SI/cudamacro.h)

add_executable(${TARGET} ${sources}) # ${headers})

#target_compile_options(${TARGET} PUBLIC -std=c++14)

target_link_libraries(${TARGET} PUBLIC ${CUDA_curand_LIBRARY})

target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/external/cub ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(${TARGET}
  PRIVATE
 #   -maxrregcount=32
  )
target_compile_options(${TARGET}
  PRIVATE
    $<$<CONFIG:DEBUG>:$<BUILD_INTERFACE:${debug_compile_flags}>>
  )
message("${CMAKE_BINARY_DIR}/x64/Debug/ALL_BUILD", "&&&&&&&&&&&")
install(TARGETS ${TARGET} DESTINATION "${CMAKE_SOURCE_DIR}/bin_SI")
#install(TARGETS ${TARGET} DESTINATION "${CMAKE_BINARY_DIR}/x64/Release/ALL_BUILD")

