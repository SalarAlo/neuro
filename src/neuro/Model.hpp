#pragma once

#include <cstddef>

#include "neuro/Dataset.hpp"

namespace Neuro {

struct TrainingConfig {
	std::size_t epochs {};
	std::size_t batch_size {};
	float learning_rate {};
	bool shuffle {};
};

template <typename Network, typename Loss>
class Model {
public:
public:
	using InputVector = typename Network::InputVector;
	using OutputVector = typename Network::OutputVector;

	OutputVector predict(const InputVector& input) {
		return m_network.forward(input);
	}

	template <Dataset D>
	void fit(const D& dataset, const TrainingConfig& config) {
		const auto count = dataset.size();

		for (std::size_t epoch {}; epoch < config.epochs; ++epoch) {
			double total_loss {};

			typename D::TargetType gradient {};
			for (std::size_t i {}; i < count; ++i) {
				const auto input = dataset.get_input(i);
				const auto target = dataset.get_target(i);

				const auto output { m_network.forward(input) };
				const auto loss { Loss::loss(output, target) };
				gradient += Loss::gradient(output, target) / static_cast<float>(config.batch_size);

				if (i % config.batch_size == 0 && i != 0) {
					m_network.backward(gradient, config.learning_rate);
					gradient.clear();
				}

				total_loss += loss;
			}

			std::println("Epoch {}: mean loss = {}", epoch + 1, total_loss / static_cast<float>(count));
		}
	}

private:
private:
	Network m_network {};
};

} // namespace Neuro
