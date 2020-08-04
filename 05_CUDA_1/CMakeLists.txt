cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(CUDA1 VERSION 1.0 LANGUAGES CXX CUDA)

#Set flags for compute-50 virtual architecture
set(CMAKE_CUDA_FLAGS "-gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_53,code=sm_53  -Wno-deprecated-gpu-targets")

#Supress warning about deprecated architectures
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

add_executable(SaxpyCudaBenchmarking SaxpyCudaBenchmarking.cu)
add_executable(LinearFilterCuda LinearFilterCuda.cu)
add_executable(HelloCuda HelloCuda.cu)

if(NOT MSVC)
	target_link_libraries(SaxpyCudaBenchmarking m)
endif()

