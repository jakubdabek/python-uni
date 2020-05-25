import functools
import operator
from typing import List, Callable, TypeVar, Iterable, Tuple

import numpy as np

from list6.activation import Sigmoid, Relu, Activation
from list6.neural_network import NeuralNetwork
from list6.utils import permutations_with_replacement


def calc_outputs(
    op: Callable[[int, int], int], inputs: Iterable[Iterable[int]]
) -> List[int]:
    return [functools.reduce(op, inp) for inp in inputs]


def main_main(size: int):
    all_inputs = np.array(permutations_with_replacement([0, 1], size))
    expected = np.array(calc_outputs(operator.xor, all_inputs)).reshape(-1, 1)

    for i in range(1):
        perm = np.random.permutation(np.arange(all_inputs.shape[0]))
        lim = len(perm) // 2
        training_set = all_inputs[perm[:lim]]
        expected_subset = expected[perm[:lim]]

        for act in permutations_with_replacement([Sigmoid(), Relu()], 2):
            for _ in range(5):
                nn = NeuralNetwork.new_random(size, [4, 1], act)
                nn.train(training_set, expected_subset, 5000)
                print(nn.predict(all_inputs[perm]))


def calc_error(expected, real) -> float:
    return np.square(expected - real).mean()


def train(
    training_set: np.ndarray,
    expected: np.ndarray,
    activations: List[Activation],
    num: int = 5,
) -> List[NeuralNetwork]:
    def make_one():
        nn = NeuralNetwork.new_random(3, [4, 1], activations)
        nn.train_trans(training_set, expected, 10000, 3)
        return nn

    return [make_one() for _ in range(num)]


def check_activation(
    data_set: np.ndarray,
    expected: np.ndarray,
    training_set_size: int,
    activations: List[Activation],
) -> Tuple[float, float]:
    training_set = data_set[:training_set_size]
    rest = data_set[training_set_size:]
    training_expected = expected[:training_set_size]
    rest_expected = expected[training_set_size:]

    networks = train(
        training_set, training_expected, activations
    )

    training_err = max(calc_error(training_expected, nn.predict_trans(training_set)) for nn in networks)
    rest_err = max(calc_error(rest_expected, nn.predict_trans(rest)) for nn in networks)

    return training_err, rest_err


def check_function(
    data_set: np.ndarray, training_set_size: int, op: Callable[[int, int], int]
) -> None:
    expected = np.array(calc_outputs(op, data_set)).reshape(-1, 1)

    for act1, act2 in permutations_with_replacement([Sigmoid(), Relu()], 2):
        print(f"{str(act1):10} {str(act2):10}")
        training_err, validation_err = check_activation(data_set, expected, training_set_size, [act1, act2])
        print(f"  training set -> {training_err}")
        print(f"validation set -> {validation_err}")


def check_training_set(data_set: np.ndarray, training_set_size: int,):
    print("-" * 15, "xor", "-" * 15)
    check_function(data_set, training_set_size, operator.xor)
    print("-" * 15, "and", "-" * 15)
    check_function(data_set, training_set_size, operator.and_)
    print("-" * 15, "or ", "-" * 15)
    check_function(data_set, training_set_size, operator.or_)


def main():
    data_set = np.array([list(map(int, bin(a)[2:].zfill(3))) for a in range(8)])
    data_set = data_set[[1, 3, 5, 7, 0, 2, 4, 6]]

    print(data_set)
    check_training_set(data_set, 4)

    data_set = np.array(permutations_with_replacement([0, 1], 3))
    np.random.shuffle(data_set)

    print(data_set)
    check_training_set(data_set, 4)


if __name__ == "__main__":
    main()
