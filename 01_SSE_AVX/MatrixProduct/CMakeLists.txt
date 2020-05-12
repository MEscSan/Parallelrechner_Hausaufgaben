cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(SSE_AVX VERSION 1.0 LANGUAGES C)

set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

add_executable(MatrixProductBenchmark MatrixProductBenchmark.c)

if(MSVC)
    target_compile_definitions(MatrixProductBenchmark PUBLIC OS_WINDOWS=1)
    target_compile_options(MatrixProductBenchmark PUBLIC "/arch:AVX2 /O2")
elseif(MINGW)
    message(WARNING "Probably not supported")
    target_compile_definitions(MatrixProductBenchmark PUBLIC OS_MINGW=1)
    target_compile_options(MatrixProductBenchmark PUBLIC "-mavx" "-O3")
    target_link_libraries(MatrixProductBenchmark m)
elseif(UNIX)
    target_compile_definitions(MatrixProductBenchmark PUBLIC OS_UNIXOID=1)
    target_compile_options(MatrixProductBenchmark PUBLIC "-mavx" "-O3")
    target_link_libraries(MatrixProductBenchmark m)
else()
    message(FATAL_ERROR "Not supported System.Compile yourself!")
endif()