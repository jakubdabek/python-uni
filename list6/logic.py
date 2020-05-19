import functools
from typing import List, Callable

import numpy as np

from list6.activation import Sigmoid, Relu
from list6.neural_network import NeuralNetwork

def generate_inputs(size: int) -> List[List[int]]:
    if size == 1:
        return [[0], [1]]
    prev = generate_inputs(size - 1)
    return [[0] + arr for arr in prev] + [[1] + arr for arr in prev]


def calc_outputs(op: Callable[[List[int]], int], inputs: List[List[int]]) -> List[int]:
    return [functools.reduce(op, inp) for inp in inputs]


def main():
    training_set = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    expected = np.array([[1], [0], [0], [1]])

    nn = NeuralNetwork.new_random(3, [5, 1], [Sigmoid(), Sigmoid()])
    nn.train_trans(training_set, expected, 10000, 5)

    print(nn.predict_trans(training_set))
    print(nn.predict_trans(np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]])))


if __name__ == "__main__":
    main()
