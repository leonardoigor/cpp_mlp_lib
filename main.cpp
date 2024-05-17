#include <iostream>
#include "NeuralNetwork.h"

int main(int, char **)
{
    std::cout << "Hello, from Test!\n";
    int inputSize = 3;
    int hiddenSize = 5;
    int outputSize = 2;

    // Create a neural network with sigmoid activation functions
    NeuralNetwork nn(inputSize, hiddenSize, outputSize, "sigmoid", "sigmoid");
    // Dummy training data: XOR problem
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 1.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 0.0},
        {1.0, 1.0, 1.0}};

    std::vector<std::vector<double>> targets = {
        {0.0, 0.0},
        {1.0, 0.0},
        {1.0, 0.0},
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {1.0, 1.0}};
    nn.printLayers();
    nn.mutation(.99);
    nn.printLayers();
    // Training the network
    double learningRate = 0.1;
    int epochs = 100000;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            nn.train(inputs[i], targets[i], learningRate);
        }
    }

    // Print the weights and biases after training
    nn.printLayers();

    // Test the network with some inputs
    std::vector<double> testInput = {1.0, 0.0, 1.0};
    std::vector<double> output = nn.feedForward(testInput);

    std::cout << "Test Input: ";
    for (const auto &val : testInput)
    {
        std::cout << val << " ";
    }
    std::cout << "\nOutput: ";
    for (const auto &val : output)
    {
        std::cout << val << " ";
    }
    std::cout << "\n";
}
