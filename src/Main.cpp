#include <print>

#include <iostream>

#include "Activations.hpp"
#include "LossFunctions.hpp"
#include "Model.hpp"
#include "NeuralNetwork.hpp"

#include "data/training_data.hpp"

int main() {
	constexpr float LEARNING_RATE { 0.01f };
	constexpr std::size_t DIGITS { 10 };
	constexpr std::size_t EPOCHS { 10 };

	TrainingDataReader reader {};

	using Network = NeuralNetwork<
	    Layer<28 * 28, 128, Relu>,
	    Layer<128, 64, Relu>,
	    Layer<64, 10, Sigmoid>>;

	Model<Network, MeanSquaredError> model {};

	const auto count { reader.get_meta_data().number_of_images };

	for (std::size_t epoch {}; epoch < EPOCHS; ++epoch) {
		double total_loss { 0.0 };

		for (std::size_t i {}; i < count; ++i) {
			const auto input { reader.get_image(i) };
			const auto label { reader.get_label(i) };

			Matrix<DIGITS, 1> target {};
			target[label, 0] = 1.0f;

			total_loss += model.train(input, target, LEARNING_RATE);
		}

		std::println(
		    "Epoch {}: mean loss = {}",
		    epoch + 1,
		    total_loss / count);
	}

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

		reader.print_image(image_index);

		const auto output { model.predict(reader.get_image(image_index)) };

		std::size_t prediction {};
		for (std::size_t digit { 1 }; digit < DIGITS; ++digit) {
			if (output[digit, 0] > output[prediction, 0])
				prediction = digit;
		}

		std::println("Prediction: {} | Actual label: {}", prediction,
		    static_cast<unsigned>(reader.get_label(image_index)));
	}
}
