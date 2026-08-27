#pragma once

#include <cstddef>
#include <random>

#include "math/Matrix.hpp"

namespace NeuralNetwork {

template <typename F, std::size_t InputSize, std::size_t OutputSize, typename Activation>
class Dense {
public:
	using Input = Matrix<F, InputSize, 1>;
	using Output = Matrix<F, OutputSize, 1>;

	Dense() {
		static std::mt19937 rnd_generator { std::random_device()() };
		const F xavier_limit = std::sqrt(F { 6 } / static_cast<F>(InputSize + OutputSize));
		std::uniform_real_distribution<F> range { -xavier_limit, xavier_limit };

		m_weights = m_weights.transform([&](auto) { return range(rnd_generator); });
	}

	Dense(const Matrix<F, OutputSize, InputSize>& weights, const Output& biases)
	    : m_weights(weights)
	    , m_biases(biases) {
	}

	Output forward(const Input& input) {
		m_input = input;
		m_pre_activation = m_weights * input + m_biases;

		return m_pre_activation.transform([this](F value) {
			return m_activation(value);
		});
	}

private:
	Input m_input {};
	Output m_pre_activation {};

	Matrix<F, OutputSize, InputSize> m_weights {};
	Matrix<F, OutputSize, 1> m_biases {};

	Activation m_activation {};
};

}
