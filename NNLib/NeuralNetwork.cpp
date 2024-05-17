#include "NeuralNetwork.h"
#include <iostream>
#include <cmath>
#include <random>
#include <stdexcept>

NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize, const std::string &hiddenActivation, const std::string &outputActivation)
    : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize),
      hiddenActivation(hiddenActivation), outputActivation(outputActivation), error(0)
{

    setup(inputSize, hiddenSize, outputSize);
}

std::vector<double> NeuralNetwork::feedForward(const std::vector<double> &inputs)
{
    std::vector<double> hiddenLayer(hiddenSize);
    std::vector<double> outputLayer(outputSize);

    for (int i = 0; i < hiddenSize; ++i)
    {
        double sum = bias1[i];
        for (int j = 0; j < inputSize; ++j)
        {
            sum += inputs[j] * weights1[j][i];
        }
        hiddenLayer[i] = activate(sum, hiddenActivation);
    }

    for (int i = 0; i < outputSize; ++i)
    {
        double sum = bias2[i];
        for (int j = 0; j < hiddenSize; ++j)
        {
            sum += hiddenLayer[j] * weights2[j][i];
        }
        outputLayer[i] = activate(sum, outputActivation);
    }

    return outputLayer;
}

void NeuralNetwork::train(const std::vector<double> &inputs, const std::vector<double> &targets, double learningRate)
{
    std::vector<double> hiddenLayer(hiddenSize);
    std::vector<double> outputLayer(outputSize);
    std::vector<double> outputErrors(outputSize);
    std::vector<double> hiddenErrors(hiddenSize);

    // Feed forward
    for (int i = 0; i < hiddenSize; ++i)
    {
        double sum = bias1[i];
        for (int j = 0; j < inputSize; ++j)
        {
            sum += inputs[j] * weights1[j][i];
        }
        hiddenLayer[i] = activate(sum, hiddenActivation);
    }

    for (int i = 0; i < outputSize; ++i)
    {
        double sum = bias2[i];
        for (int j = 0; j < hiddenSize; ++j)
        {
            sum += hiddenLayer[j] * weights2[j][i];
        }
        outputLayer[i] = activate(sum, outputActivation);
    }

    // Backpropagation
    for (int i = 0; i < outputSize; ++i)
    {
        outputErrors[i] = targets[i] - outputLayer[i];
    }

    std::vector<double> outputDelta(outputSize);
    for (int i = 0; i < outputSize; ++i)
    {
        outputDelta[i] = outputErrors[i] * activateDerivative(outputLayer[i], outputActivation);
    }

    std::vector<double> hiddenDelta(hiddenSize);
    for (int i = 0; i < hiddenSize; ++i)
    {
        double sum = 0;
        for (int j = 0; j < outputSize; ++j)
        {
            sum += outputDelta[j] * weights2[i][j];
        }
        hiddenDelta[i] = sum * activateDerivative(hiddenLayer[i], hiddenActivation);
    }

    // Update weights and biases
    for (int i = 0; i < inputSize; ++i)
    {
        for (int j = 0; j < hiddenSize; ++j)
        {
            weights1[i][j] += learningRate * hiddenDelta[j] * inputs[i];
        }
    }

    for (int i = 0; i < hiddenSize; ++i)
    {
        for (int j = 0; j < outputSize; ++j)
        {
            weights2[i][j] += learningRate * outputDelta[j] * hiddenLayer[i];
        }
    }

    for (int i = 0; i < hiddenSize; ++i)
    {
        bias1[i] += learningRate * hiddenDelta[i];
    }

    for (int i = 0; i < outputSize; ++i)
    {
        bias2[i] += learningRate * outputDelta[i];
    }

    error = 0;
    for (const auto &e : hiddenDelta)
    {
        error += e;
    }
}

void NeuralNetwork::mutation(double mutateRate)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    for (int i = 0; i < hiddenSize; ++i)
    {
        for (int j = 0; j < outputSize; ++j)
        {
            if (dis(gen) < mutateRate)
            {
                weights2[i][j] = dis(gen) * 2000 - 1000;
            }
        }
    }

    for (int i = 0; i < outputSize; ++i)
    {
        if (dis(gen) < mutateRate)
        {
            bias2[i] = dis(gen) * 2000 - 1000;
        }
    }

    if (dis(gen) < mutateRate)
    {
        hiddenSize++;
        setup(inputSize, hiddenSize, outputSize);
    }

    if (dis(gen) < mutateRate)
    {
        int neuronIndex = (int)std::floor(dis(gen) * hiddenSize);
        int weightIndex = (int)std::floor(dis(gen) * weights2[neuronIndex].size());
        weights2[neuronIndex][weightIndex] = dis(gen) * 2000 - 1000;
    }
}

void NeuralNetwork::printLayers() const
{
    std::cout << "Input Layer Weights:\n";
    for (int i = 0; i < inputSize; ++i)
    {
        std::cout << "Neuron " << i << ":";
        for (const auto &weight : weights1[i])
        {
            std::cout << " " << weight;
        }
        std::cout << "\n";
    }
    std::cout << "Input Layer Biases:";
    for (const auto &bias : bias1)
    {
        std::cout << " " << bias;
    }
    std::cout << "\n";

    std::cout << "Hidden Layer Weights:\n";
    for (int i = 0; i < hiddenSize; ++i)
    {
        std::cout << "Neuron " << i << ":";
        for (const auto &weight : weights2[i])
        {
            std::cout << " " << weight;
        }
        std::cout << "\n";
    }
    std::cout << "Hidden Layer Biases:";
    for (const auto &bias : bias2)
    {
        std::cout << " " << bias;
    }
    std::cout << "\n";
}

void NeuralNetwork::setup(int inputSize, int hiddenSize, int outputSize)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1000, 1000);
    std::vector<std::vector<double>> newWeights1(inputSize, std::vector<double>(hiddenSize));
    weights1 = newWeights1;
    weights2.resize(hiddenSize, std::vector<double>(outputSize));
    bias1.resize(hiddenSize);
    bias2.resize(outputSize);
    for (int i = 0; i < inputSize; ++i)
    {
        for (int j = 0; j < hiddenSize; ++j)
        {
            weights1[i][j] = dis(gen);
        }
    }

    for (int i = 0; i < hiddenSize; ++i)
    {
        for (int j = 0; j < outputSize; ++j)
        {
            weights2[i][j] = dis(gen);
        }
    }

    for (int i = 0; i < hiddenSize; ++i)
    {
        bias1[i] = dis(gen);
    }

    for (int i = 0; i < outputSize; ++i)
    {
        bias2[i] = dis(gen);
    }
}

double NeuralNetwork::activate(double x, const std::string &activation) const
{
    if (activation == "sigmoid")
    {
        return 1 / (1 + std::exp(-x));
    }
    else if (activation == "relu")
    {
        return x > 0 ? x : 0;
    }
    else if (activation == "leakyRelu")
    {
        return x > 0 ? x : 0.1 * x;
    }
    else if (activation == "linear")
    {
        return x;
    }
    else
    {
        throw std::invalid_argument("Unknown activation function: " + activation);
    }
}

double NeuralNetwork::activateDerivative(double x, const std::string &activation) const
{
    if (activation == "sigmoid")
    {
        return x * (1 - x);
    }
    else if (activation == "relu")
    {
        return x > 0 ? 1 : 0;
    }
    else if (activation == "leakyRelu")
    {
        return x > 0 ? 1 : 0.1;
    }
    else if (activation == "linear")
    {
        return 1;
    }
    else
    {
        throw std::invalid_argument("Unknown activation function: " + activation);
    }
}

double NeuralNetwork::random() const
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0, 1);
    return dis(gen);
}
