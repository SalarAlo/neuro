#pragma once

#include <cmath>

#include "activation/ActivationFunction.hpp"

namespace NeuralNetwork {

template <typename T>
class Sigmoid : public IActivationFunction<T> {
public:
	T operator()(T x) const override {
		return 1 / (1 + std::exp(-x));
	}

	T derivative(T x) const override {
		return std::exp(-x) / std::pow((1 + std::exp(-x)), 2);
	}
};

}
