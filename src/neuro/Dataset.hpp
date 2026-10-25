#pragma once

#include <algorithm>
#include <concepts>

#include "neuro/Matrix.hpp"
#include "neuro/TrainingData.hpp"

namespace Neuro {

template <typename T>
concept Dataset = requires(const T& dataset, std::size_t index) {
	typename T::InputType;
	typename T::TargetType;

	{ dataset.size() } -> std::convertible_to<std::size_t>;

	{ dataset.get_input(index) } -> std::same_as<typename T::InputType>;
	{ dataset.get_target(index) } -> std::same_as<typename T::TargetType>;
};

class MNISTDataset {
public:
	using InputType = Matrix<28 * 28, 1>;
	using TargetType = Matrix<10, 1>;

	explicit MNISTDataset(const TrainingDataReader& reader, std::size_t sample_limit = 0)
	    : m_reader { reader }
	    , m_sample_limit { sample_limit } { }

	std::size_t size() const {
		const auto available { static_cast<std::size_t>(m_reader.get_train_meta_data().number_of_images) };
		return m_sample_limit == 0 ? available : std::min(available, m_sample_limit);
	}

	InputType get_input(std::size_t index) const {
		return m_reader.get_train_image(index);
	}

	TargetType get_target(std::size_t index) const {
		TargetType target {};

		const auto label = m_reader.get_train_label(index);
		target[label, 0] = 1.0f;

		return target;
	}

private:
	const TrainingDataReader& m_reader;
	std::size_t m_sample_limit;
};

}
