cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

# Add the subdirectory containing the NeuralNetwork library
add_subdirectory(NNLib)

# Add the executable target
add_executable(APP main.cpp)

# Link the NeuralNetwork library to the executable
target_link_libraries(APP PUBLIC NeuralNetwork)

# Link the mylib library to the executable if needed
target_link_libraries(APP PUBLIC mylib)

# Set the output directory for the executable
set_target_properties(APP PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
