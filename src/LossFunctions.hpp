#include <cstddef>

#include "Matrix.hpp"

struct MeanSquaredError {

	template <std::size_t N>
	static float loss(const Matrix<N, 1>& output, const Matrix<N, 1>& target) {
		const auto difference { output - target };

		float loss {};

		difference.for_every([&loss](float d) {
			loss += d * d;
		});

		return loss / (2.0f * static_cast<float>(N));
	}

	template <std::size_t N>
	static Matrix<N, 1> gradient(const Matrix<N, 1>& output, const Matrix<N, 1>& target) {
		return (output - target) / static_cast<float>(N);
	}
};
