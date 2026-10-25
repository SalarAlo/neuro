#include <print>

#include <charconv>
#include <cstddef>
#include <iostream>
#include <optional>
#include <string_view>

#include "neuro/Activations.hpp"
#include "neuro/LossFunctions.hpp"
#include "neuro/Model.hpp"
#include "neuro/NeuralNetwork.hpp"
#include "neuro/TrainingData.hpp"

namespace {

std::optional<std::size_t> parse_positive(std::string_view text) {
	std::size_t value {};
	const auto [end, error] { std::from_chars(text.data(), text.data() + text.size(), value) };
	if (error != std::errc {} || end != text.data() + text.size() || value == 0)
		return std::nullopt;
	return value;
}

}

int main(int argc, char** argv) {
	std::size_t sample_limit { 1000 };
	std::size_t epochs { 5 };
	bool no_prompt {};

	for (int i { 1 }; i < argc; ++i) {
		const std::string_view option { argv[i] };
		if (option == "--help") {
			std::println("Usage: neuro [--samples N] [--epochs N] [--no-prompt]");
			std::println("  --samples N   Train on the first N images (default: 1000; up to 60000).");
			std::println("  --epochs N    Number of training epochs (default: 5).");
			std::println("  --no-prompt   Exit after training.");
			return 0;
		}
		if (option == "--no-prompt") {
			no_prompt = true;
			continue;
		}
		if (option == "--samples" || option == "--epochs") {
			if (++i >= argc) {
				std::println(stderr, "Missing value for {}.", option);
				return 2;
			}
			const auto value { parse_positive(argv[i]) };
			if (!value) {
				std::println(stderr, "{} requires a positive integer.", option);
				return 2;
			}
			(option == "--samples" ? sample_limit : epochs) = *value;
			continue;
		}
		std::println(stderr, "Unknown option: {}. Use --help for usage.", option);
		return 2;
	}

	Neuro::TrainingDataReader reader {};
	const Neuro::MNISTDataset dataset { reader, sample_limit };

	using Network = Neuro::NeuralNetwork<
	    Neuro::Layer<28 * 28, 64, Neuro::Relu>,
	    Neuro::Layer<64, 32, Neuro::Relu>,
	    Neuro::Layer<32, 10, Neuro::Sigmoid>>;

	Neuro::Model<Network, Neuro::MeanSquaredError> model {};

	Neuro::TrainingConfig config {
		.epochs = epochs,
		.batch_size = 100,
		.learning_rate = 0.01f,
	};

	std::println("Training on {} images for {} epochs.", dataset.size(), epochs);
	model.fit(dataset, config);
	if (no_prompt)
		return 0;

	const auto count { reader.get_test_meta_data().number_of_images };

	for (;;) {
		std::print("Enter image index (0–{}), or -1 to quit: ", count - 1);
		std::cout << std::flush;

		long long index {};
		if (!(std::cin >> index) || index == -1)
			break;

		if (index < 0 || static_cast<std::size_t>(index) >= count) {
			std::println("Invalid index.");
			continue;
		}

		const auto image_index { static_cast<std::size_t>(index) };

		reader.print_test_image(image_index);

		const auto output { model.predict(reader.get_test_image(image_index)) };

		std::size_t prediction {};
		for (std::size_t digit { 0 }; digit < Neuro::MNISTDataset::TargetType::ROWS_AMOUNT; ++digit) {
			if (output[digit, 0] > output[prediction, 0])
				prediction = digit;
		}

		std::println("Prediction: {} | Actual label: {}", prediction, static_cast<unsigned>(reader.get_test_label(image_index)));
	}
}
