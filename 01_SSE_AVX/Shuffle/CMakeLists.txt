cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(Shuffle_Test VERSION 1.0 LANGUAGES C)

# set some compiler options
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

add_executable(ShuffleTest ShuffleTest.c)


if(MSVC)
  target_compile_definitions(ShuffleTest PUBLIC OS_WINDOWS=1)
elseif(MINGW)
  message(WARNING "Probably not supported")
  target_compile_definitions(ShuffleTest PUBLIC OS_MINGW=1)
  target_link_libraries(ShuffleTest m)
elseif(UNIX)
  target_compile_definitions(ShuffleTest PUBLIC OS_UNIXOID=1)
  target_link_libraries(ShuffleTest m)
else()
  message(FATAL_ERROR "No Supported System. Compile Yourself!")
endif()
