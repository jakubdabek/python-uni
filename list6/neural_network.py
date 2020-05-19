from typing import List
from dataclasses import dataclass

import numpy as np

from list6.activation import Activation


@dataclass
class Layer:
    input_weights: np.ndarray
    biases: np.ndarray
    activation: Activation

    @property
    def size(self) -> int:
        return self.input_weights.shape[0]

    def calculate(self, values: np.ndarray) -> np.ndarray:
        """Calculates layer outputs

        Arguments:
            values: Samples x InputSize array

        Returns:
            Samples x NeuronsCount array of outputs
        """
        return self.activation.value(np.dot(self.input_weights, values) + self.biases)


class NeuralNetwork:
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    @classmethod
    def new_random(
        cls, input_count: int, neuron_counts: List[int], activations: List[Activation]
    ):
        layers = []
        rng = np.random.default_rng()
        for neuron_count, activation in zip(neuron_counts, activations):
            layers.append(
                Layer(
                    rng.standard_normal((neuron_count, input_count)),
                    np.zeros((neuron_count, 1)),
                    activation,
                )
            )
            input_count = neuron_count

        return cls(layers)

    def feed_forward(self, input_values: np.ndarray) -> List[np.ndarray]:
        """Feeds the input through the network

        Args:
            input_values: column vectors with inputs (input_size x samples)

        Returns:
            list of activation values as column vectors for each input (layer_size x samples)
        """
        output = input_values  # input is treated like output of "layer 0"
        layer_outputs = [output]
        for layer in self.layers:
            output = layer.calculate(output)
            layer_outputs.append(output)

        return layer_outputs

    def feed_forward_trans(self, input_values: np.ndarray) -> List[np.ndarray]:
        return [output.T for output in self.feed_forward(input_values.T)]

    def back_propagation(
        self,
        input_values: np.ndarray,
        expected_outputs: np.ndarray,
        learning_rate: float,
    ) -> None:
        outputs = self.feed_forward(input_values)

        output_layer = self.layers[-1]
        output_delta = (
            outputs[-1] - expected_outputs
        ) * output_layer.activation.derivative_from_value(outputs[-1])

        changes = []

        last_delta = output_delta
        last_layer = output_layer
        for layer, output in zip(self.layers[-2::-1], outputs[-2::-1]):
            changes.append(
                (
                    last_delta.mean(axis=1, keepdims=True),
                    np.einsum("ij,kj->jik", last_delta, output).mean(axis=0),
                )
            )
            last_delta = np.dot(
                last_layer.input_weights.T, last_delta
            ) * layer.activation.derivative_from_value(output)
            last_layer = layer

        changes.append(
            (
                last_delta.mean(axis=1, keepdims=True),
                np.einsum("ij,kj->jik", last_delta, outputs[0]).mean(axis=0),
            )
        )

        for layer, (bias_change, weight_change) in zip(self.layers, reversed(changes)):
            layer.input_weights -= learning_rate * weight_change
            layer.biases -= learning_rate * bias_change

    def train(
        self, input_set: np.ndarray, expected_outputs: np.ndarray, epochs: int, learning_rate: float = 0.5
    ) -> None:
        for _ in range(epochs):
            self.back_propagation(input_set, expected_outputs, learning_rate)

    def train_trans(
        self, input_set: np.ndarray, expected_outputs: np.ndarray, epochs: int, learning_rate: float = 0.5
    ) -> None:
        return self.train(input_set.T, expected_outputs.T, epochs, learning_rate)

    def predict(self, input_set: np.ndarray) -> np.ndarray:
        return self.feed_forward(input_set)[-1]

    def predict_trans(self, input_set: np.ndarray) -> np.ndarray:
        return self.feed_forward_trans(input_set)[-1]
