import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Tuple, Callable, Optional, List, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn import preprocessing
from pygifsicle import optimize

from list6.activation import *
from list6.neural_network import NeuralNetwork


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

    fig.show()

    line: plt.Line2D
    (line,) = ax.plot(
        data.testing_input,
        data.y_scaler.inverse_transform(nn.predict_trans(data.testing_input_scaled)),
        "o",
    )
    title: plt.Text = ax.set_title("0 iterations")

    # scaled_expected = y_scaler.inverse_transform(expected)
    # ylim = np.min(scaled_expected), np.max(scaled_expected)
    # ax.set_ylim(*ylim)

    fig.show()

    single_epoch, iters = nn_params.epochs

    def update(i):
        i += 1
        nn.train_trans(
            data.training_input_scaled, data.training_expected_scaled, single_epoch, 1
        )
        output = data.y_scaler.inverse_transform(
            nn.predict_trans(data.testing_input_scaled)
        )
        line.set_ydata(output)
        error = calc_error(data.testing_expected, output)
        title.set_text(f"{i * single_epoch} iterations (error: {error:.5})")
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
    training_input: np.ndarray,
    testing_input: np.ndarray,
    nn_params: NeuralNetworkParams,
    gif_params: Optional[GifParams],
):
    expected = f(training_input)
    plt.scatter(training_input, expected)
    plt.show()

    x_scaler = Scaler()
    x_scaler.fit(training_input)

    y_scaler = Scaler()
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


def main_square(gif_params: GifParams = None):
    return check_function(
        np.square,
        np.linspace(-50, 50, 26).reshape(-1, 1),
        np.linspace(-50, 50, 101).reshape(-1, 1),
        NeuralNetworkParams(list(zip([10, 1], [Tanh(), Tanh()])), (10000, 20)),
        gif_params,
    )


def main_sin(gif_params: GifParams = None):
    return check_function(
        lambda x: np.sin((3 * np.pi / 2) * x),
        np.linspace(0, 2, 21).reshape(-1, 1),
        np.linspace(0, 2, 161).reshape(-1, 1),
        NeuralNetworkParams(list(zip([10, 1], [Tanh(), Tanh()])), (10000, 40)),
        gif_params,
    )


def main():
    _square_filename = "plot-square-1.gif"
    main_square(None)
    _sin_filename = "plot-sin-1.gif"
    main_sin(None)


if __name__ == "__main__":
    main()
