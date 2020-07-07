cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(CUDA1 VERSION 1.0 LANGUAGES C CUDA)

add_executable(SaxpyCudaBenchmarking SaxpyCudaBenchmarking.cu)

if(NOT MSVC)
	target_link_libraries(SaxpyCudaBenchmarking m)
endif()

