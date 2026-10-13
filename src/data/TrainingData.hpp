#pragma once
#include <cstdint>

#include <fstream>
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

struct TrainingData {
	TrainingData() = default;
	TrainingData(std::ifstream& image_file_train, std::ifstream& label_file_train);

	std::vector<std::uint8_t> bytes {};
	std::vector<std::uint8_t> labels {};
	TrainingMetaData meta_data {};
};

static_assert(sizeof(TrainingMetaData) == 16);

class TrainingDataReader {
public:
	TrainingDataReader();

	const TrainingMetaData& get_train_meta_data() const;
	Matrix<28 * 28, 1> get_train_image(std::size_t index) const;
	std::uint8_t get_train_label(std::size_t index) const;

	void print_train_image(std::size_t index) const;

	const TrainingMetaData& get_test_meta_data() const;
	Matrix<28 * 28, 1> get_test_image(std::size_t index) const;
	std::uint8_t get_test_label(std::size_t index) const;

	void print_test_image(std::size_t index) const;

private:
	TrainingData m_train_data {};
	TrainingData m_test_data {};

	std::vector<std::uint8_t> m_test_bytes {};
	std::vector<std::uint8_t> m_test_labels {};
};
