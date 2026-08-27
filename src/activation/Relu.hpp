#pragma once

#include "activation/ActivationFunction.hpp"

namespace NeuralNetwork {

template <typename T>
class Relu : public IActivationFunction<T> {
public:
	T operator()(T x) const override {
		return x < 0 ? 0 : x;
	}

	T derivative(T x) const override {
		return x < 0 ? 0 : 1;
	}
};

}
