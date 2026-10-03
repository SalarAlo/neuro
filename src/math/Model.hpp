template <typename Network, typename Loss>
class Model {
public:
	using InputVector = typename Network::InputVector;
	using OutputVector = typename Network::OutputVector;

	OutputVector predict(const InputVector& input) {
		return m_network.forward(input);
	}

	float train(const InputVector& input, const OutputVector& target, float learning_rate) {
		const auto output { m_network.forward(input) };
		const auto loss { Loss::loss(output, target) };
		const auto gradient { Loss::gradient(output, target) };

		m_network.backward(gradient, learning_rate);

		return loss;
	}

private:
	Network m_network {};
};
