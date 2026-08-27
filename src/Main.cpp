#include <print>

#include "Dense.hpp"

#include "activation/Identity.hpp"

#include "math/Matrix.hpp"

int main() {
	NeuralNetwork::Matrix<float, 1, 2> weights {};
	weights[0, 0] = 2;
	weights[0, 1] = 3;

	NeuralNetwork::Matrix<float, 1, 1> biases {};
	biases[0, 0] = 1;

	NeuralNetwork::Matrix<float, 2, 1> input {};
	input[0, 0] = 4;
	input[1, 0] = 5;

	NeuralNetwork::Dense<float, 2, 1, NeuralNetwork::Identity<float>> dense { weights, biases };
	NeuralNetwork::Matrix<float, 1, 1> out { dense.forward(input) };

	std::println("{}", out[0, 0]);

	return 0;
}
