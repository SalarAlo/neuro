#pragma once

#include <cmath>

#include <algorithm>

namespace Neuro {

struct Identity {
	float operator()(float x) {
		return x;
	}

	float derivative(float) {
		return 1.0f;
	}
};

struct Relu {
	float operator()(float x) {
		return std::max(x, 0.0f);
	}

	float derivative(float x) {
		return x > 0.0f ? 1.0f : 0.0f;
	}
};

struct Sigmoid {
	float operator()(float x) {
		return 1 / (1 + std::exp(-x));
	}

	float derivative(float x) {
		float y { operator()(x) };
		return y * (1.0f - y);
	}
};

}
