#include <cstring>
#include <print>

#include <bit>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "TrainingData.hpp"

TrainingData::TrainingData(std::ifstream& image_file_train, std::ifstream& label_file_train) {
	if (!image_file_train)
		throw std::runtime_error("Could not open file.");

	bytes = std::vector<std::uint8_t>(
	    std::istreambuf_iterator<char> { image_file_train },
	    std::istreambuf_iterator<char> {});

	labels = std::vector<std::uint8_t>(
	    std::istreambuf_iterator<char> { label_file_train },
	    std::istreambuf_iterator<char> {});

	if (bytes.size() < sizeof(TrainingMetaData))
		throw std::runtime_error("File is too small.");

	meta_data = TrainingMetaData { bytes.data() };
}

TrainingMetaData::TrainingMetaData(void* bytes) {
	std::memcpy(this, bytes, sizeof(TrainingMetaData));

	magic_number = std::byteswap(magic_number);
	number_of_images = std::byteswap(number_of_images);
	rows = std::byteswap(rows);
	cols = std::byteswap(cols);
}

TrainingDataReader::TrainingDataReader() {
	std::ifstream image_file_train { "./data/train-images.idx3-ubyte", std::ios::binary };
	std::ifstream label_file_train { "./data/train-labels.idx1-ubyte", std::ios::binary };
	m_train_data = TrainingData { image_file_train, label_file_train };

	std::ifstream image_file_test { "./data/t10k-images.idx3-ubyte", std::ios::binary };
	std::ifstream label_file_test { "./data/t10k-labels.idx1-ubyte", std::ios::binary };
	m_test_data = TrainingData { image_file_test, label_file_test };
}

const TrainingMetaData& TrainingDataReader::get_train_meta_data() const {
	return m_train_data.meta_data;
}

Matrix<28 * 28, 1> TrainingDataReader::get_train_image(std::size_t index) const {
	if (index >= m_train_data.meta_data.number_of_images)
		throw std::out_of_range("Image index out of range.");

	const std::size_t header_size { sizeof(TrainingMetaData) };
	const std::size_t rows { m_train_data.meta_data.rows };
	const std::size_t cols { m_train_data.meta_data.cols };
	const std::size_t available { m_train_data.bytes.size() - header_size };

	const std::size_t image_size { rows * cols };

	if (index >= available / image_size)
		throw std::runtime_error("Incomplete image data.");

	const std::size_t offset { header_size + index * image_size };
	Matrix<28 * 28, 1> m {};
	auto img { std::span<const std::uint8_t> { m_train_data.bytes }.subspan(offset, image_size) };
	for (std::size_t row {}; row < rows; row++) {
		for (std::size_t col {}; col < cols; col++) {
			m[row * cols + col, 0] = static_cast<float>(img[row * cols + col] / 255.0f);
		}
	}

	return m;
}

std::uint8_t TrainingDataReader::get_train_label(std::size_t index) const {
	return m_train_data.labels[8 + index];
}

void TrainingDataReader::print_train_image(std::size_t index) const {
#pragma region BRIGHTNESS
	std::vector<char> brightness {
		'@',
		'$',
		'B',
		'%',
		'8',
		'&',
		'W',
		'M',
		'*',
		'o',
		'a',
		'h',
		'k',
		'b',
		'd',
		'p',
		'q',
		'w',
		'm',
		'Z',
		'O',
		'0',
		'Q',
		'L',
		'C',
		'J',
		'U',
		'Y',
		'X',
		'z',
		'c',
		'v',
		'u',
		'n',
		'x',
		'r',
		'j',
		'f',
		't',
		'/',
		'(',
		')',
		'1',
		'{',
		'[',
		']',
		'?',
		'-',
		'_',
		'+',
		'~',
		'<',
		'>',
		'i',
		'!',
		'l',
		'I',
		';',
		':',
		',',
		'"',
		'`',
		'\'',
		'.'

	};
#pragma endregion

	auto image = get_train_image(index);

	for (std::size_t row = 0; row < m_train_data.meta_data.rows; ++row) {
		for (std::size_t col = 0; col < m_train_data.meta_data.cols; ++col) {
			const auto pixel = static_cast<unsigned char>(image[row * m_train_data.meta_data.cols + col, 0] * 255.0f);

			const std::size_t brightness_idx { (255u - pixel) * (brightness.size() - 1) / 255u };

			const char symbol = pixel == 0 ? ' ' : brightness[brightness_idx];

			std::print("{}{}", symbol, symbol);
		}
		std::println();
	}
	std::println("This is an image of a {}", get_train_label(index));
}

const TrainingMetaData& TrainingDataReader::get_test_meta_data() const {
	return m_test_data.meta_data;
}

Matrix<28 * 28, 1> TrainingDataReader::get_test_image(std::size_t index) const {
	if (index >= m_test_data.meta_data.number_of_images)
		throw std::out_of_range("Image index out of range.");

	const std::size_t header_size { sizeof(TrainingMetaData) };
	const std::size_t rows { m_test_data.meta_data.rows };
	const std::size_t cols { m_test_data.meta_data.cols };
	const std::size_t available { m_test_data.bytes.size() - header_size };

	const std::size_t image_size { rows * cols };

	if (index >= available / image_size)
		throw std::runtime_error("Incomplete image data.");

	const std::size_t offset { header_size + index * image_size };
	Matrix<28 * 28, 1> m {};
	auto img { std::span<const std::uint8_t> { m_test_data.bytes }.subspan(offset, image_size) };
	for (std::size_t row {}; row < rows; row++) {
		for (std::size_t col {}; col < cols; col++) {
			m[row * cols + col, 0] = static_cast<float>(img[row * cols + col] / 255.0f);
		}
	}

	return m;
}

std::uint8_t TrainingDataReader::get_test_label(std::size_t index) const {
	return m_test_data.labels[8 + index];
}

void TrainingDataReader::print_test_image(std::size_t index) const {
#pragma region BRIGHTNESS
	std::vector<char> brightness {
		'@',
		'$',
		'B',
		'%',
		'8',
		'&',
		'W',
		'M',
		'*',
		'o',
		'a',
		'h',
		'k',
		'b',
		'd',
		'p',
		'q',
		'w',
		'm',
		'Z',
		'O',
		'0',
		'Q',
		'L',
		'C',
		'J',
		'U',
		'Y',
		'X',
		'z',
		'c',
		'v',
		'u',
		'n',
		'x',
		'r',
		'j',
		'f',
		't',
		'/',
		'(',
		')',
		'1',
		'{',
		'[',
		']',
		'?',
		'-',
		'_',
		'+',
		'~',
		'<',
		'>',
		'i',
		'!',
		'l',
		'I',
		';',
		':',
		',',
		'"',
		'`',
		'\'',
		'.'

	};
#pragma endregion

	auto image = get_test_image(index);

	for (std::size_t row = 0; row < m_test_data.meta_data.rows; ++row) {
		for (std::size_t col = 0; col < m_test_data.meta_data.cols; ++col) {
			const auto pixel = static_cast<unsigned char>(image[row * m_test_data.meta_data.cols + col, 0] * 255.0f);

			const std::size_t brightness_idx { (255u - pixel) * (brightness.size() - 1) / 255u };

			const char symbol = pixel == 0 ? ' ' : brightness[brightness_idx];

			std::print("{}{}", symbol, symbol);
		}
		std::println();
	}
	std::println("This is an image of a {}", get_test_label(index));
}
