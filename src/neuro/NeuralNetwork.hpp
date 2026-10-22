#pragma once

#include <cstddef>
#include <tuple>

#include "neuro/Layer.hpp"

namespace Neuro {

template <LayerConcept InputLayer, LayerConcept... Layers>
        requires NetworkLayerArrangement<InputLayer, Layers...>
class NeuralNetwork {
private:
	using LayerTuple = std::tuple<InputLayer, Layers...>;

public:
	static constexpr std::size_t LayersAmount = sizeof...(Layers) + 1;

	using InputVector = typename InputLayer::InputVector;
	using OutputVector =
	    typename std::tuple_element_t<
	        LayersAmount - 1, LayerTuple>::OutputVector;

	template <std::size_t I = 0, typename Input = InputVector>
	auto forward(const Input& input) {
		if constexpr (I < LayersAmount) {
			auto output { std::get<I>(m_layers).forward(input) };
			return forward<I + 1>(output);
		} else {
			return input;
		}
	}

	template <std::size_t I = LayersAmount, typename Gradient = OutputVector>
	auto backward(const Gradient& gradient) {
		if constexpr (I > 0) {
			auto input_gradient {
				std::get<I - 1>(m_layers).backward(gradient)
			};

			return backward<I - 1>(input_gradient);
		} else {
			return gradient;
		}
	}

	void step(float learning_rate) {
		std::apply([learning_rate](auto&... layers) {
			(layers.step(learning_rate), ...);
		},
		    m_layers);
	}

	void zero_grad() {
		std::apply([](auto&... layers) {
			(layers.zero_grad(), ...);
		},
		    m_layers);
	}

private:
	LayerTuple m_layers {};
};

}
