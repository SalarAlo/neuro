#pragma once

namespace NeuralNetwork {

template <typename T>
class IActivationFunction {
public:
	virtual T operator()(T x) const = 0;
	virtual T derivative(T x) const = 0;
};

}
