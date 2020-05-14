from typing import List
from dataclasses import dataclass

import numpy as np

from list6.activation import Activation


@dataclass
class Layer:
    input_weights: np.ndarray
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
        return self.activation.value(np.dot(values, self.input_weights))


class NeuralNetwork:
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    @classmethod
    def new_random(
        cls, input_count: int, neuron_counts: List[int], activations: List[Activation]
    ):
        layers = []
        for neuron_count, activation in zip(neuron_counts, activations):
            layers.append(
                Layer(np.random.random_sample((input_count, neuron_count)), activation)
            )
            input_count = neuron_count

        return cls(layers)

    def feed_forward(self, input_values: np.ndarray) -> List[np.ndarray]:
        layer_outputs = []
        output = input_values  # input is treated like output of "layer 0"
        for layer in self.layers:
            output = layer.calculate(output)
            layer_outputs.append(output)

        return layer_outputs

    def back_propagation(
        self,
        input_values: np.ndarray,
        expected_outputs: np.ndarray,
        learning_rate: float,
    ) -> None:
        if len(self.layers) != 2:
            raise NotImplementedError(
                "backpropagation implemented only for simple perceptrons (2 layers)"
            )

        outputs = self.feed_forward(input_values)

        activation1 = self.layers[-1].activation
        error1 = (expected_outputs - outputs[-1]) * activation1.gradient_from_value(outputs[-1])
        delta1 = learning_rate * np.dot(error1.T, outputs[-2])

        activation2 = self.layers[-2].activation
        error2 = activation2.gradient_from_value(outputs[-2]) * np.dot(
            error1, self.layers[-1].input_weights.T
        )
        delta2 = learning_rate * np.dot(error2.T, input_values)

        self.layers[-1].input_weights += delta1.T
        self.layers[-2].input_weights += delta2.T

    def train(
        self, input_set: np.ndarray, expected_outputs: np.ndarray, epochs: int
    ) -> None:
        for _ in range(epochs):
            self.back_propagation(input_set, expected_outputs, 0.5)

    def predict(self, input_set: np.ndarray):
        return self.feed_forward(input_set)
