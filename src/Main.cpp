#include <print>

#include <iostream>

#include "neuro/Activations.hpp"
#include "neuro/LossFunctions.hpp"
#include "neuro/Model.hpp"
#include "neuro/NeuralNetwork.hpp"
#include "neuro/TrainingData.hpp"

int main() {
	Neuro::TrainingDataReader reader {};

	using Network = Neuro::NeuralNetwork<
	    Neuro::Layer<28 * 28, 128, Neuro::Relu>,
	    Neuro::Layer<128, 64, Neuro::Relu>,
	    Neuro::Layer<64, 10, Neuro::Sigmoid>>;

	Neuro::Model<Network, Neuro::MeanSquaredError> model {};

	auto count { reader.get_train_meta_data().number_of_images };

	Neuro::TrainingConfig config {
		.epochs = 3,
		.batch_size = 1,
		.learning_rate = 0.01f,
	};

	model.fit(Neuro::MNISTDataset { reader }, config);

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
		for (std::size_t digit { 0 }; digit < Neuro::MNISTDataset::TargetType::get_rows(); ++digit) {
			if (output[digit, 0] > output[prediction, 0])
				prediction = digit;
		}

		std::println("Prediction: {} | Actual label: {}", prediction, static_cast<unsigned>(reader.get_test_label(image_index)));
	}
}
