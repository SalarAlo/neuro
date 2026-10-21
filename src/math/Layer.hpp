#pragma once

#include <cmath>

#include <cstddef>
#include <random>

#include "Matrix.hpp"

namespace Neuro {

template <std::size_t InputNeurons, std::size_t OutputNeurons, typename ActivationFn>
class Layer {
public:
	using InputVector = Matrix<InputNeurons, 1>;
	using OutputVector = Matrix<OutputNeurons, 1>;

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

	InputVector backward(const OutputVector& output_gradient, float learning_rate) {
		InputVector input_gradient {};

		for (std::size_t row {}; row < OutputNeurons; ++row) {
			const float error { output_gradient[row, 0] * m_activation_fn.derivative(m_last_z[row, 0]) };
			const float step { learning_rate * error };

			for (std::size_t col {}; col < InputNeurons; ++col) {
				float& weight = m_weights[row, col];

				input_gradient[col, 0] += weight * error;

				weight -= step * m_last_input[col, 0];
			}

			m_biases[row, 0] -= step;
		}

		return input_gradient;
	}

private:
	InputVector m_last_input {};
	OutputVector m_last_z {};

	OutputVector m_biases {};
	Matrix<OutputNeurons, InputNeurons> m_weights {};

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

} // namespace Neuro
