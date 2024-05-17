# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)
set(dir ${CMAKE_CURRENT_SOURCE_DIR}/library)

# Define the library
add_library(mylib STATIC mylib.cpp)

# Specify include directories
target_include_directories(mylib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
