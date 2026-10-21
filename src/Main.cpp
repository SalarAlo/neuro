#include <print>

#include <iostream>

#include "Activations.hpp"
#include "LossFunctions.hpp"
#include "Model.hpp"
#include "NeuralNetwork.hpp"

#include "data/TrainingData.hpp"

int main() {
	TrainingDataReader reader {};

	using Network = NeuralNetwork<
	    Layer<28 * 28, 128, Relu>,
	    Layer<128, 64, Relu>,
	    Layer<64, 10, Sigmoid>>;

	Model<Network, MeanSquaredError> model {};

	auto count { reader.get_train_meta_data().number_of_images };

	TrainingConfig config {
		.epochs = 3,
		.batch_size = 1,
		.learning_rate = 0.01f,
	};

	model.fit(MNISTDataset { reader }, config);

	count = reader.get_test_meta_data().number_of_images;
	for (;;) {
		std::print("Enter image index (0–{}), or -1 to quit: ", count);
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
		for (std::size_t digit { 0 }; digit < MNISTDataset::TargetType::get_rows(); ++digit) {
			if (output[digit, 0] > output[prediction, 0])
				prediction = digit;
		}

		std::println("Prediction: {} | Actual label: {}", prediction, static_cast<unsigned>(reader.get_test_label(image_index)));
	}
}
