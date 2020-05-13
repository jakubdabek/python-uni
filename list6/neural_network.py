from typing import Callable, TypeVar, List, Tuple
from dataclasses import dataclass

import numpy as np

ActivationInput = TypeVar("ActivationInput", float, np.ndarray)
ActivationFunction = Callable[[ActivationInput], ActivationInput]
ActivationGradient = Callable[[ActivationInput], ActivationInput]
Activation = Tuple[ActivationFunction, ActivationGradient]


def sigmoid(x: ActivationInput) -> ActivationInput:
    return 1.0 / (1.0 + np.exp(np.negative(x)))


def sigmoid_gradient(x: ActivationInput) -> ActivationInput:
    fx = sigmoid(x)
    return fx * (1.0 - fx)


SIGMOID_ACTIVATION: Activation = (sigmoid, sigmoid_gradient)


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
        return self.activation[0](np.dot(values, self.input_weights))


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
        activation: Activation,
        learning_rate: float,
    ) -> None:
        if len(self.layers) != 2:
            raise NotImplementedError(
                "backpropagation implemented only for simple perceptrons (2 layers)"
            )

        activation_func, activation_grad = activation
        outputs = self.feed_forward(input_values)

        error1 = (expected_outputs - outputs[-1]) * activation_grad(outputs[-1])
        delta1 = learning_rate * np.dot(error1.T, outputs[-2])
        error2 = activation_grad(outputs[-2]) * np.dot(
            error1, self.layers[-1].input_weights.T
        )
        delta2 = learning_rate * np.dot(error2.T, input_values)

        self.layers[-1].input_weights += delta1.T
        self.layers[-2].input_weights += delta2.T

    def train(
        self, input_set: np.ndarray, expected_outputs: np.ndarray, epochs: int
    ) -> None:
        for _ in range(epochs):
            self.back_propagation(input_set, expected_outputs, SIGMOID_ACTIVATION, 0.5)

    def predict(self, input_set: np.ndarray):
        return self.feed_forward(input_set)
