#pragma once

#include <cmath>

#include <cstddef>
#include <random>

#include "neuro/Matrix.hpp"

namespace Neuro {

template <std::size_t InputNeurons, std::size_t OutputNeurons, typename ActivationFn>
class Layer {
public:
	using InputVector = Matrix<InputNeurons, 1>;
	using OutputVector = Matrix<OutputNeurons, 1>;
	using WeightsMatrix = Matrix<OutputNeurons, InputNeurons>;

	constexpr static std::size_t InputNeuronsAmount { InputNeurons };
	constexpr static std::size_t OutputNeuronsAmount { OutputNeurons };

public:
	Layer() {
		std::mt19937 rng { std::random_device {}() };
		std::normal_distribution<float> dist {
			0.0f,
			std::sqrt(2.0f / InputNeurons)
		};

		for (std::size_t row {}; row < OutputNeurons; row++) {
			for (std::size_t col {}; col < InputNeurons; col++) {
				m_weights[row, col] = dist(rng);
			}
		}
	}

	OutputVector forward(const InputVector& input) {
		m_last_input = input;
		m_last_z = (m_weights * input) + m_biases;
		return m_last_z.apply(m_activation_fn);
	}

	InputVector backward(const OutputVector& output_gradient) {
		InputVector input_gradient {};

		for (std::size_t row {}; row < OutputNeurons; ++row) {
			const float error { output_gradient[row, 0] * m_activation_fn.derivative(m_last_z[row, 0]) };

			for (std::size_t col {}; col < InputNeurons; ++col) {
				input_gradient[col, 0] += m_weights[row, col] * error;
				m_weight_gradients[row, col] += error * m_last_input[col, 0];
			}

			m_bias_gradients[row, 0] += error;
		}

		return input_gradient;
	}

	void step(float learning_rate) {
		for (std::size_t row {}; row < OutputNeurons; ++row) {
			for (std::size_t col {}; col < InputNeurons; ++col)
				m_weights[row, col] -= learning_rate * m_weight_gradients[row, col];

			m_biases[row, 0] -= learning_rate * m_bias_gradients[row, 0];
		}
	}

	void zero_grad() {
		m_weight_gradients.clear();
		m_bias_gradients.clear();
	}

private:
	InputVector m_last_input {};
	OutputVector m_last_z {};

	OutputVector m_biases {};
	WeightsMatrix m_weights {};
	OutputVector m_bias_gradients {};
	WeightsMatrix m_weight_gradients {};

	ActivationFn m_activation_fn {};
};

template <typename T>
struct IsLayerClass
    : std::false_type { };
template <std::size_t InputNeurons, std::size_t OutputNeurons, typename ActivationFn>
struct IsLayerClass<Layer<InputNeurons, OutputNeurons, ActivationFn>>
    : std::true_type { };

template <typename T>
concept LayerConcept = IsLayerClass<T>::value;

template <typename... Ts>
concept NetworkLayerArrangement = requires {
	requires(LayerConcept<Ts> && ...);
	requires(sizeof...(Ts) >= 1);
	requires([]<std::size_t... I>(std::index_sequence<I...>) {
		using Layers = std::tuple<Ts...>;

		return ((
		            std::tuple_element_t<I, Layers>::OutputNeuronsAmount == std::tuple_element_t<I + 1, Layers>::InputNeuronsAmount)
		    && ...);
	}(std::make_index_sequence<sizeof...(Ts) - 1> {}));
};

}
