#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>

namespace NeuralNetwork {

template <typename T, std::size_t Rows, std::size_t Cols>
class Matrix {
public:
#pragma region ACCESS
	[[nodiscard]]
	T& operator[](std::size_t row, std::size_t col) {
		return m_matrix[col + row * Cols];
	}

	[[nodiscard]]
	const T& operator[](std::size_t row, std::size_t col) const {
		return m_matrix[col + row * Cols];
	}

	[[nodiscard]]
	const T& at(std::size_t row, std::size_t col) const {
		const auto idx { col + row * Cols };
		if (idx >= Rows * Cols) {
			throw std::out_of_range("Matrix Access beyond limits");
		}
		return m_matrix[idx];
	}

	[[nodiscard]]
	T& at(std::size_t row, std::size_t col) {
		const auto idx { col + row * Cols };
		if (idx >= Rows * Cols) {
			throw std::out_of_range("Matrix Access beyond limits");
		}
		return m_matrix[idx];
	}

#pragma endregion

#pragma region OPERATORS
	[[nodiscard]]
	Matrix operator+(const Matrix& other) const {
		Matrix m {};
		for (std::size_t rows {}; rows < Rows; rows++) {
			for (std::size_t cols {}; cols < Cols; cols++) {
				m.at(rows, cols) = other.at(rows, cols) + at(rows, cols);
			}
		}
		return m;
	}

	[[nodiscard]]
	Matrix operator-(const Matrix& other) const {
		Matrix m {};
		for (std::size_t rows {}; rows < Rows; rows++) {
			for (std::size_t cols {}; cols < Cols; cols++) {
				m.at(rows, cols) = at(rows, cols) - other.at(rows, cols);
			}
		}
		return m;
	}

	[[nodiscard]]
	Matrix operator*(T value) const {
		Matrix m {};
		for (std::size_t rows {}; rows < Rows; rows++) {
			for (std::size_t cols {}; cols < Cols; cols++) {
				m.at(rows, cols) = at(rows, cols) * value;
			}
		}
		return m;
	}

	template <std::size_t OtherCols>
	[[nodiscard]]
	Matrix<T, Rows, OtherCols> operator*(const Matrix<T, Cols, OtherCols>& other) const {
		Matrix<T, Rows, OtherCols> m {};
		for (std::size_t y {}; y < Rows; y++) {
			for (std::size_t x {}; x < OtherCols; x++) {
				for (std::size_t k {}; k < Cols; k++) {
					m.at(y, x) += at(y, k) * other.at(k, x);
				}
			}
		}
		return m;
	}

	[[nodiscard]]
	Matrix<T, Cols, Rows> transpose() const {
		Matrix<T, Cols, Rows> m {};
		for (std::size_t row {}; row < Cols; row++) {
			for (std::size_t col {}; col < Rows; col++) {
				m.at(row, col) = at(col, row);
			}
		}
		return m;
	}

	[[nodiscard]]
	Matrix hadamard(const Matrix& other) const {
		Matrix m {};
		for (std::size_t row {}; row < Rows; row++) {
			for (std::size_t col {}; col < Cols; col++) {
				m.at(row, col) = at(row, col) * other.at(row, col);
			}
		}
		return m;
	}

	[[nodiscard]]
	Matrix transform(std::function<T(T value)> f) const {
		Matrix m {};

		for (std::size_t row {}; row < Rows; row++) {
			for (std::size_t col {}; col < Cols; col++) {
				m.at(row, col) = f(at(row, col));
			}
		}

		return m;
	}
#pragma endregion

private:
	std::array<T, Rows * Cols> m_matrix {};
};

}
