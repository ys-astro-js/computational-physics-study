import math
import random


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def sigmoid_derivative(value):
    return value * (1.0 - value)


def predict(x1, x2, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    hidden_outputs = []

    for hidden_index in range(2):
        total = (
            x1 * weights_input_hidden[0][hidden_index]
            + x2 * weights_input_hidden[1][hidden_index]
            + bias_hidden[hidden_index]
        )
        hidden_outputs.append(sigmoid(total))

    output_total = bias_output
    for hidden_index in range(2):
        output_total += hidden_outputs[hidden_index] * weights_hidden_output[hidden_index]

    return sigmoid(output_total), hidden_outputs


def loss(data, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    total_loss = 0.0

    for x1, x2, target in data:
        output, _ = predict(
            x1,
            x2,
            weights_input_hidden,
            bias_hidden,
            weights_hidden_output,
            bias_output,
        )
        total_loss += (target - output) ** 2

    return total_loss / len(data)


def train_xor_network(learning_rate=0.5, epochs=10000):
    training_data = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
    ]
    validation_data = training_data

    random.seed(7)
    weights_input_hidden = [
        [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)],
        [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)],
    ]
    bias_hidden = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
    weights_hidden_output = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
    bias_output = random.uniform(-1.0, 1.0)
    history = []

    for epoch in range(1, epochs + 1):
        for x1, x2, target in training_data:
            output, hidden_outputs = predict(
                x1,
                x2,
                weights_input_hidden,
                bias_hidden,
                weights_hidden_output,
                bias_output,
            )

            output_error = target - output
            output_delta = output_error * sigmoid_derivative(output)

            hidden_deltas = []
            for hidden_index in range(2):
                hidden_error = output_delta * weights_hidden_output[hidden_index]
                hidden_deltas.append(
                    hidden_error * sigmoid_derivative(hidden_outputs[hidden_index])
                )

            for hidden_index in range(2):
                weights_hidden_output[hidden_index] += (
                    learning_rate * output_delta * hidden_outputs[hidden_index]
                )
            bias_output += learning_rate * output_delta

            for hidden_index in range(2):
                weights_input_hidden[0][hidden_index] += (
                    learning_rate * hidden_deltas[hidden_index] * x1
                )
                weights_input_hidden[1][hidden_index] += (
                    learning_rate * hidden_deltas[hidden_index] * x2
                )
                bias_hidden[hidden_index] += learning_rate * hidden_deltas[hidden_index]

        if epoch % 1000 == 0:
            train_loss = loss(
                training_data,
                weights_input_hidden,
                bias_hidden,
                weights_hidden_output,
                bias_output,
            )
            validation_loss = loss(
                validation_data,
                weights_input_hidden,
                bias_hidden,
                weights_hidden_output,
                bias_output,
            )
            history.append((epoch, train_loss, validation_loss))

            if train_loss < 0.01 and validation_loss < 0.01:
                break

    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, history


def main():
    (
        weights_input_hidden,
        bias_hidden,
        weights_hidden_output,
        bias_output,
        history,
    ) = train_xor_network()

    print("XOR neural network")
    print("\nepoch | train loss | validation loss")
    print("-" * 38)

    for epoch, train_loss, validation_loss in history:
        print(f"{epoch:>5} | {train_loss:>10.4f} | {validation_loss:>15.4f}")

    print("\nx1 x2 | output | y")
    print("-" * 20)

    for x1, x2, target in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        output, _ = predict(
            x1,
            x2,
            weights_input_hidden,
            bias_hidden,
            weights_hidden_output,
            bias_output,
        )
        prediction = 1 if output >= 0.5 else 0
        print(f"{x1:>2} {x2:>2} | {output:>6.3f} | {prediction}")


if __name__ == "__main__":
    main()
