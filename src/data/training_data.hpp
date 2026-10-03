#pragma once
#include <cstdint>

#include <span>
#include <vector>

#include "Matrix.hpp"

struct TrainingMetaData {
	std::uint32_t magic_number {};
	std::uint32_t number_of_images {};
	std::uint32_t rows {};
	std::uint32_t cols {};

	TrainingMetaData() = default;
	TrainingMetaData(void* bytes);
};

static_assert(sizeof(TrainingMetaData) == 16);

class TrainingDataReader {
public:
	TrainingDataReader();

	const TrainingMetaData& get_meta_data() const;
	Matrix<28 * 28, 1> get_image(std::size_t index) const;
	std::uint8_t get_label(std::size_t index) const;

	void print_image(std::size_t index) const;

private:
	std::vector<std::uint8_t> m_training_bytes {};
	std::vector<std::uint8_t> m_labels {};
	TrainingMetaData m_meta_data {};
};
