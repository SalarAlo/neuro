#pragma once

#include <print>

#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>

template <std::size_t Rows, std::size_t Cols>
struct Matrix {
public:
	Matrix() = default;

	Matrix(std::array<std::array<float, Cols>, Rows> values) {
		for (std::size_t row {}; row < Rows; ++row) {
			for (std::size_t col {}; col < Cols; ++col) {
				m_array[row * Cols + col] = values[row][col];
			}
		}
	}

	const float& get(std::size_t row, std::size_t col) const {
		return m_array[row * Cols + col];
	}

	float& get(std::size_t row, std::size_t col) {
		return m_array[row * Cols + col];
	}

	std::size_t get_rows() const { return Rows; }
	std::size_t get_cols() const { return Cols; }

	float& operator[](std::size_t row, std::size_t col) {
		return m_array[row * Cols + col];
	}

	const float& operator[](std::size_t row, std::size_t col) const {
		return m_array[row * Cols + col];
	}

	Matrix operator+(const Matrix& other) const {
		Matrix result {};

		for (std::size_t i {}; i < Rows * Cols; ++i) {
			result.m_array[i] = m_array[i] + other.m_array[i];
		}

		return result;
	}

	void operator+=(const Matrix& other) {
		*this = *this + other;
	}

	Matrix operator-(const Matrix& other) const {
		Matrix result {};

		for (std::size_t i {}; i < Rows * Cols; ++i) {
			result.m_array[i] = m_array[i] - other.m_array[i];
		}

		return result;
	}

	template <std::size_t OtherRows, std::size_t OtherCols>
	Matrix<Rows, OtherCols> operator*(
	    const Matrix<OtherRows, OtherCols>& other) const {
		static_assert(
		    Cols == OtherRows,
		    "Incompatible dimensions: left columns must equal right rows.");

		Matrix<Rows, OtherCols> result {};

		if constexpr (OtherCols == 1) {
			for (std::size_t row {}; row < Rows; ++row) {
				float sum {};

				for (std::size_t k {}; k < Cols; ++k) {
					sum += m_array[row * Cols + k] * other[k, 0];
				}

				result[row, 0] = sum;
			}
		} else {
			for (std::size_t row {}; row < Rows; ++row) {
				for (std::size_t k {}; k < Cols; ++k) {
					const float value = m_array[row * Cols + k];

					for (std::size_t col {}; col < OtherCols; ++col) {
						result[row, col] += value * other[k, col];
					}
				}
			}
		}

		return result;
	}

	Matrix operator*(float scalar) const {
		Matrix result {};

		for (std::size_t i {}; i < Rows * Cols; ++i) {
			result.m_array[i] = m_array[i] * scalar;
		}

		return result;
	}

	Matrix operator/(float scalar) const {
		return operator*(1.0f / scalar);
	}

	template <typename Fn>
	        requires(
	            std::is_invocable_r_v<
	                float, Fn&, std::size_t, std::size_t>
	            || std::is_invocable_r_v<float, Fn&, float>)
	Matrix apply(Fn&& fn) const {
		Matrix result {};

		if constexpr (
		    std::is_invocable_r_v<
		        float, Fn&, std::size_t, std::size_t>) {
			for (std::size_t row {}; row < Rows; ++row) {
				for (std::size_t col {}; col < Cols; ++col) {
					result.m_array[row * Cols + col] = std::invoke(fn, row, col);
				}
			}
		} else {
			for (std::size_t i {}; i < Rows * Cols; ++i) {
				result.m_array[i] = std::invoke(fn, m_array[i]);
			}
		}

		return result;
	}

	template <typename Fn>
	        requires(
	            std::is_invocable_r_v<
	                void, Fn&, std::size_t, std::size_t>
	            || std::is_invocable_r_v<void, Fn&, float>)
	void for_every(Fn&& fn) const {
		if constexpr (
		    std::is_invocable_r_v<
		        float, Fn&, std::size_t, std::size_t>) {
			for (std::size_t row {}; row < Rows; ++row) {
				for (std::size_t col {}; col < Cols; ++col) {
					std::invoke(fn, row, col);
				}
			}
		} else {
			for (std::size_t i {}; i < Rows * Cols; ++i) {
				std::invoke(fn, m_array[i]);
			}
		}
	}

	Matrix<Cols, Rows> transpose() const {
		Matrix<Cols, Rows> result {};

		for (std::size_t row {}; row < Rows; ++row) {
			for (std::size_t col {}; col < Cols; ++col) {
				result[col, row] = m_array[row * Cols + col];
			}
		}

		return result;
	}

	void clear() {
		*this = *this - *this;
	}

	void print() const {
		std::println("MATRIX {}X{}", Rows, Cols);

		for (std::size_t row {}; row < Rows; ++row) {
			std::print("[ ");

			for (std::size_t col {}; col < Cols; ++col) {
				std::print(
				    "{:>5}{}",
				    get(row, col),
				    col + 1 < Cols ? ", " : "");
			}

			std::println(" ]");
		}

		std::println("__________________");
	}

private:
	std::array<float, Rows * Cols> m_array {};
};
