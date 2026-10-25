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
		const auto count { dataset.size() };

		for (std::size_t epoch {}; epoch < config.epochs; ++epoch) {
			double total_loss {};

			for (std::size_t i {}; i < count; ++i) {
				const auto input { dataset.get_input(i) };
				const auto target { dataset.get_target(i) };

				const auto output { m_network.forward(input) };

				total_loss += Loss::loss(output, target);

				const auto gradient { Loss::gradient(output, target) };

				m_network.backward(gradient);

				const bool batch_complete { (i + 1) % config.batch_size == 0 };

				const bool last_sample { i + 1 == count };

				if (batch_complete || last_sample) {
					const auto batch_count { (i % config.batch_size) + 1 };

					m_network.step(config.learning_rate / static_cast<float>(batch_count));
					m_network.zero_grad();
				}
			}

			std::println(
			    "Epoch {}: mean loss = {}",
			    epoch + 1,
			    total_loss / static_cast<double>(count));
		}
	}

private:
private:
	Network m_network {};
};

}
