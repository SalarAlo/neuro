#include <print>

#include <iterator>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "neuro/TrainingData.hpp"

namespace Neuro {

namespace {
constexpr std::size_t image_header_size { 16 };
constexpr std::size_t label_header_size { 8 };
constexpr std::size_t image_size { 28 * 28 };

std::uint32_t read_big_endian_u32(const std::uint8_t* bytes) {
	return (static_cast<std::uint32_t>(bytes[0]) << 24)
	    | (static_cast<std::uint32_t>(bytes[1]) << 16)
	    | (static_cast<std::uint32_t>(bytes[2]) << 8)
	    | static_cast<std::uint32_t>(bytes[3]);
}

std::vector<std::uint8_t> read_file(std::ifstream& file) {
	if (!file)
		throw std::runtime_error("Could not open data file.");

	std::vector<std::uint8_t> result {
		std::istreambuf_iterator<char> { file },
		std::istreambuf_iterator<char> {}
	};
	if (file.bad())
		throw std::runtime_error("Could not read data file.");
	return result;
}

TrainingData load_data(const char* image_path, const char* label_path) {
	std::ifstream images { image_path, std::ios::binary };
	std::ifstream labels { label_path, std::ios::binary };
	return TrainingData { images, labels };
}
} // namespace

TrainingMetaData::TrainingMetaData(const std::uint8_t* bytes)
    : magic_number { read_big_endian_u32(bytes) }
    , number_of_images { read_big_endian_u32(bytes + 4) }
    , rows { read_big_endian_u32(bytes + 8) }
    , cols { read_big_endian_u32(bytes + 12) } { }

TrainingData::TrainingData(std::ifstream& image_file, std::ifstream& label_file)
    : bytes { read_file(image_file) }
    , labels { read_file(label_file) } {
	if (bytes.size() < image_header_size || labels.size() < label_header_size)
		throw std::runtime_error("Incomplete IDX header.");

	meta_data = TrainingMetaData { bytes.data() };
	const auto label_count { read_big_endian_u32(labels.data() + 4) };
	if (meta_data.magic_number != 2051 || read_big_endian_u32(labels.data()) != 2049)
		throw std::runtime_error("Invalid IDX file type.");
	if (meta_data.rows != 28 || meta_data.cols != 28)
		throw std::runtime_error("Expected 28 by 28 images.");
	if (label_count != meta_data.number_of_images)
		throw std::runtime_error("Image and label counts differ.");

	const auto count { static_cast<std::size_t>(meta_data.number_of_images) };
	if (count > (std::numeric_limits<std::size_t>::max() - image_header_size) / image_size
	    || count > std::numeric_limits<std::size_t>::max() - label_header_size
	    || bytes.size() != image_header_size + count * image_size
	    || labels.size() != label_header_size + count)
		throw std::runtime_error("Incomplete or oversized IDX data.");
}

const TrainingMetaData& TrainingData::get_meta_data() const {
	return meta_data;
}

Matrix<28 * 28, 1> TrainingData::get_image(std::size_t index) const {
	if (index >= meta_data.number_of_images)
		throw std::out_of_range("Image index out of range.");

	const auto offset { image_header_size + index * image_size };
	Matrix<28 * 28, 1> image {};
	for (std::size_t pixel {}; pixel < image_size; ++pixel)
		image[pixel, 0] = static_cast<float>(bytes[offset + pixel]) / 255.0f;
	return image;
}

std::uint8_t TrainingData::get_label(std::size_t index) const {
	if (index >= meta_data.number_of_images)
		throw std::out_of_range("Label index out of range.");
	return labels[label_header_size + index];
}

void TrainingData::print_image(std::size_t index) const {
	constexpr std::string_view brightness {
		"@$B%8&WM*oahkbdpqwmZO0QLCJUYXzcvunxrjft/()1{[]?-_+~<>i!lI;:,\"`'."
	};
	const auto image { get_image(index) };

	for (std::size_t row {}; row < meta_data.rows; ++row) {
		for (std::size_t col {}; col < meta_data.cols; ++col) {
			const auto pixel { static_cast<unsigned char>(image[row * meta_data.cols + col, 0] * 255.0f) };
			const auto brightness_index { (255u - pixel) * (brightness.size() - 1) / 255u };
			const char symbol { pixel == 0 ? ' ' : brightness[brightness_index] };
			std::print("{}{}", symbol, symbol);
		}
		std::println();
	}
	std::println("This is an image of a {}", get_label(index));
}

TrainingDataReader::TrainingDataReader()
    : m_train_data { load_data("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte") }
    , m_test_data { load_data("./data/t10k-images.idx3-ubyte", "./data/t10k-labels.idx1-ubyte") } { }

const TrainingMetaData& TrainingDataReader::get_train_meta_data() const {
	return m_train_data.get_meta_data();
}

Matrix<28 * 28, 1> TrainingDataReader::get_train_image(std::size_t index) const {
	return m_train_data.get_image(index);
}

std::uint8_t TrainingDataReader::get_train_label(std::size_t index) const {
	return m_train_data.get_label(index);
}

void TrainingDataReader::print_train_image(std::size_t index) const {
	m_train_data.print_image(index);
}

const TrainingMetaData& TrainingDataReader::get_test_meta_data() const {
	return m_test_data.get_meta_data();
}

Matrix<28 * 28, 1> TrainingDataReader::get_test_image(std::size_t index) const {
	return m_test_data.get_image(index);
}

std::uint8_t TrainingDataReader::get_test_label(std::size_t index) const {
	return m_test_data.get_label(index);
}

void TrainingDataReader::print_test_image(std::size_t index) const {
	m_test_data.print_image(index);
}

} // namespace Neuro
