cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(MPI VERSION 1.0 LANGUAGES C)

if(APPLE)
	message(WARNING "Apple is just supported with external compilers")
endif()

find_package(MPI REQUIRED)
#find_package(OpenSSL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)


message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMp_EXE_LINKER_FLAGS}")

add_executable(MonteCarloPi MonteCarloPi.c)
target_link_libraries(MonteCarloPi MPI::MPI_C)

add_executable(Workload Workload.c)
target_link_libraries(Workload MPI::MPI_C)
target_link_libraries(Workload m)