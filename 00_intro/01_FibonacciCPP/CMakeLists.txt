cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(01_FibonacciCPP VERSION 1.0 LANGUAGES CXX)

add_executable(Fibonacci fibonacci.cpp)

if(MSVC)
 message("Hello from the MSVC compiler. Nothing to do here")
else()
    target_link_libraries(Fibonacci m)
endif()