#pragma once

#include "activation/ActivationFunction.hpp"

namespace NeuralNetwork {
template <typename T>
class Identity : public IActivationFunction<T> {
public:
	T operator()(T x) const override {
		return x;
	}

	T derivative(T x) const override {
		return 1;
	}
};

}
