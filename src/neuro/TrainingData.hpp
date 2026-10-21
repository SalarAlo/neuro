#pragma once
#include <cstdint>

#include <cstddef>
#include <fstream>
#include <vector>

#include "neuro/Matrix.hpp"

namespace Neuro {

struct TrainingMetaData {
	std::uint32_t magic_number {};
	std::uint32_t number_of_images {};
	std::uint32_t rows {};
	std::uint32_t cols {};

	TrainingMetaData() = default;
	TrainingMetaData(const std::uint8_t* bytes);
};

class TrainingData {
public:
	TrainingData(std::ifstream& image_file, std::ifstream& label_file);

	const TrainingMetaData& get_meta_data() const;
	Matrix<28 * 28, 1> get_image(std::size_t index) const;
	std::uint8_t get_label(std::size_t index) const;
	void print_image(std::size_t index) const;

private:
	std::vector<std::uint8_t> bytes {};
	std::vector<std::uint8_t> labels {};
	TrainingMetaData meta_data {};
};

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
	TrainingData m_train_data;
	TrainingData m_test_data;
};

}
