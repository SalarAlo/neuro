#include <cstring>
#include <print>

#include <bit>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "training_data.hpp"

TrainingMetaData::TrainingMetaData(void* bytes) {
	std::memcpy(this, bytes, sizeof(TrainingMetaData));

	magic_number = std::byteswap(magic_number);
	number_of_images = std::byteswap(number_of_images);
	rows = std::byteswap(rows);
	cols = std::byteswap(cols);
}

TrainingDataReader::TrainingDataReader() {
	std::ifstream image_file { "./data/train-images.idx3-ubyte", std::ios::binary };
	std::ifstream label_file { "./data/train-labels.idx1-ubyte", std::ios::binary };

	if (!image_file)
		throw std::runtime_error("Could not open file.");

	// Store the bytes in the member variable.
	m_training_bytes = std::vector<std::uint8_t>(
	    std::istreambuf_iterator<char> { image_file },
	    std::istreambuf_iterator<char> {});

	m_labels = std::vector<std::uint8_t>(
	    std::istreambuf_iterator<char> { label_file },
	    std::istreambuf_iterator<char> {});

	if (m_training_bytes.size() < sizeof(TrainingMetaData))
		throw std::runtime_error("File is too small.");

	m_meta_data = TrainingMetaData { m_training_bytes.data() };
}

const TrainingMetaData& TrainingDataReader::get_meta_data() const {
	return m_meta_data;
}

Matrix<28 * 28, 1> TrainingDataReader::get_image(std::size_t index) const {
	if (index >= m_meta_data.number_of_images)
		throw std::out_of_range("Image index out of range.");

	const std::size_t header_size { sizeof(TrainingMetaData) };
	const std::size_t rows { m_meta_data.rows };
	const std::size_t cols { m_meta_data.cols };
	const std::size_t available { m_training_bytes.size() - header_size };

	const std::size_t image_size { rows * cols };

	if (index >= available / image_size)
		throw std::runtime_error("Incomplete image data.");

	const std::size_t offset { header_size + index * image_size };
	Matrix<28 * 28, 1> m {};
	auto img { std::span<const std::uint8_t> { m_training_bytes }.subspan(offset, image_size) };
	for (std::size_t row {}; row < rows; row++) {
		for (std::size_t col {}; col < cols; col++) {
			m[row * cols + col, 0] = static_cast<float>(img[row * cols + col] / 255.0f);
		}
	}

	return m;
}

std::uint8_t TrainingDataReader::get_label(std::size_t index) const {
	return m_labels[8 + index];
}

void TrainingDataReader::print_image(std::size_t index) const {
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

	auto image = get_image(index);

	for (std::size_t row = 0; row < m_meta_data.rows; ++row) {
		for (std::size_t col = 0; col < m_meta_data.cols; ++col) {
			const auto pixel = static_cast<unsigned char>(image[row * m_meta_data.cols + col, 0] * 255.0f);

			const std::size_t brightness_idx { (255u - pixel) * (brightness.size() - 1) / 255u };

			const char symbol = pixel == 0 ? ' ' : brightness[brightness_idx];

			std::print("{}{}", symbol, symbol);
		}
		std::println();
	}
	std::println("This is an image of a {}", get_label(index));
}
