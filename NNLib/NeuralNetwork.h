#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>

class NeuralNetwork
{
public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize, const std::string &hiddenActivation, const std::string &outputActivation);

    std::vector<double> feedForward(const std::vector<double> &inputs);
    void train(const std::vector<double> &inputs, const std::vector<double> &targets, double learningRate);
    void mutation(double mutateRate);
    void printLayers() const;

private:
    int inputSize;
    int hiddenSize;
    int outputSize;
    double error;
    std::string hiddenActivation;
    std::string outputActivation;
    std::vector<std::vector<double>> weights1;
    std::vector<std::vector<double>> weights2;
    std::vector<double> bias1;
    std::vector<double> bias2;

    void setup(int inputSize, int hiddenSize, int outputSize);
    double activate(double x, const std::string &activation) const;
    double activateDerivative(double x, const std::string &activation) const;
    double random() const;
};

#endif // NEURALNETWORK_H
