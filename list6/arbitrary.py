import logging
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import Tuple, Callable, Optional, List, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn import preprocessing
from pygifsicle import optimize

from list6.activation import *
from list6.neural_network import NeuralNetwork
from list6.utils import permutations_with_replacement


def calc_error(expected, real) -> float:
    return np.square(expected - real).mean()


Scaler = preprocessing.MinMaxScaler


@dataclass
class Data:
    training_data: Tuple[np.ndarray, np.ndarray]
    testing_data: Tuple[np.ndarray, np.ndarray]
    scalers: Tuple[Scaler, Scaler]

    @property
    def x_scaler(self):
        return self.scalers[0]

    @property
    def y_scaler(self):
        return self.scalers[1]

    @property
    def training_input(self):
        return self.training_data[0]

    @property
    def training_expected(self):
        return self.training_data[1]

    @property
    def testing_input(self):
        return self.testing_data[0]

    @property
    def testing_expected(self):
        return self.testing_data[1]

    @cached_property
    def training_input_scaled(self):
        return self.x_scaler.transform(self.training_input)

    @cached_property
    def training_expected_scaled(self):
        return self.y_scaler.transform(self.training_expected)

    @cached_property
    def testing_input_scaled(self):
        return self.x_scaler.transform(self.testing_input)

    @cached_property
    def testing_expected_scaled(self):
        return self.y_scaler.transform(self.testing_expected)


@dataclass
class NeuralNetworkParams:
    layers: List[Tuple[int, Activation]]
    epochs: Tuple[int, int]


@dataclass
class GifParams:
    filename: Union[str, Path]
    interval: int = 100


def train_progress(
    data: Data, nn_params: NeuralNetworkParams, gif_params: Optional[GifParams],
):
    nn = NeuralNetwork.new_random(1, nn_params.layers)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    ax.plot(
        data.testing_input, data.testing_expected,
    )

    nn_params_text = " ".join(
        f"({l_size}-{l_act})" for l_size, l_act in nn_params.layers
    )
    title_template = f"{nn_params_text}\n{{iters}} iterations (error: {{error:.5}})"

    untrained_output = data.y_scaler.inverse_transform(
        nn.predict_trans(data.testing_input_scaled)
    )
    untrained_error = calc_error(data.testing_expected, untrained_output)
    line: plt.Line2D
    (line,) = ax.plot(data.testing_input, untrained_output, "o",)
    title: plt.Text = ax.set_title(
        title_template.format(iters=0, error=untrained_error)
    )

    # scaled_expected = y_scaler.inverse_transform(expected)
    # ylim = np.min(scaled_expected), np.max(scaled_expected)
    # ax.set_ylim(*ylim)

    fig.show()

    single_epoch, iters = nn_params.epochs

    def update(i):
        i += 1
        nn.train_trans(
            data.training_input_scaled, data.training_expected_scaled, single_epoch
        )
        output = data.y_scaler.inverse_transform(
            nn.predict_trans(data.testing_input_scaled)
        )
        line.set_ydata(output)
        error = calc_error(data.testing_expected, output)
        title.set_text(title_template.format(iters=i * single_epoch, error=error))
        # print(i, *[layer.input_weights for layer in nn.layers], sep="\n")
        print(i, error)
        # ax.relim()
        # ax.autoscale_view()
        return line, title

    if gif_params is not None:
        anim = FuncAnimation(
            fig,
            update,
            init_func=lambda: (line, title),
            frames=iters,
            interval=gif_params.interval,
            blit=True,
        )
        anim.save(gif_params.filename, dpi=100, writer="imagemagick")
        try:
            optimize(gif_params.filename)
        except FileNotFoundError:
            logging.warning("gifsicle not installed")
    else:
        for i in range(iters):
            update(i)
        ax.plot(
            data.testing_input, data.testing_expected,
        )
        fig.show()


def check_function(
    f: Callable[[np.ndarray], np.ndarray],
    input_data: Tuple[np.ndarray, np.ndarray],
    nn_params: NeuralNetworkParams,
    gif_params: Optional[GifParams],
):
    training_input, testing_input = input_data
    expected = f(training_input)
    # plt.scatter(training_input, expected)
    # plt.show()

    x_scaler = Scaler()
    x_scaler.fit(training_input)

    if isinstance(nn_params.layers[-1][1], Tanh):
        y_bounds = (-1, 1)
    else:
        y_bounds = (0, 1)

    y_scaler = Scaler(y_bounds)
    y_scaler.fit(expected)

    testing_expected = f(testing_input)

    train_progress(
        Data(
            (training_input, expected),
            (testing_input, testing_expected),
            (x_scaler, y_scaler),
        ),
        nn_params,
        gif_params,
    )
    return training_input, expected


def main_square(filename_template: Optional[str] = None):
    return main_any(
        np.square,
        (
            np.linspace(-50, 50, 26).reshape(-1, 1),
            np.linspace(-50, 50, 101).reshape(-1, 1),
        ),
        filename_template,
    )


def main_sin(filename_template: Optional[str] = None):
    return main_any(
        lambda x: np.sin((3 * np.pi / 2) * x),
        (np.linspace(0, 2, 21).reshape(-1, 1), np.linspace(0, 2, 161).reshape(-1, 1)),
        filename_template,
    )


def main_any(
    f: Callable[[np.ndarray], np.ndarray],
    data: Tuple[np.ndarray, np.ndarray],
    filename_template: Optional[str],
):
    def perms(num: int):
        return permutations_with_replacement([Relu(), Sigmoid(), Tanh()], num)

    for activations in chain(perms(2), perms(3)):
        params: Optional[GifParams] = None
        if filename_template is not None:
            filename = filename_template.format(
                activations="-".join(map(str, activations)), suffix="1"
            )
            params = GifParams(filename)

        neuron_counts = [10] * len(activations)
        neuron_counts[-1] = 1
        check_function(
            f,
            data,
            NeuralNetworkParams(list(zip(neuron_counts, activations)), (2000, 300)),
            params,
        )


def main():
    _square_filename = "plots/plot-square-{activations}-{suffix}.gif"
    main_square(_square_filename)
    _sin_filename = "plots/plot-sin-{activations}-{suffix}.gif"
    main_sin(_sin_filename)


if __name__ == "__main__":
    main()
