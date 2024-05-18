# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Create the NeuralNetwork library
add_library(NeuralNetwork SHARED NeuralNetwork.cpp)

# Specify include directories for the NeuralNetwork library
target_include_directories(NeuralNetwork PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Add the subdirectory containing the mylib library
add_subdirectory(library)

# Include directories for the executable
target_include_directories(NeuralNetwork PUBLIC library/mylib)
target_link_libraries(NeuralNetwork PUBLIC mylib)

# Define export macro for Windows
target_compile_definitions(NeuralNetwork PRIVATE NeuralNetwork_EXPORTS)