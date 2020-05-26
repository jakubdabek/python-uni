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

from list4.time_decorator import Timer
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
) -> List[float]:
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

    errors = []
    error_checks = 5
    error_check_iters = [iters, iters - 1] + [iters // error_checks * i for i in reversed(range(1, error_checks))]

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

        if error_check_iters and error_check_iters[-1] == i:
            error_check_iters.pop()
            errors.append(error)

        # print(i, *[layer.input_weights for layer in nn.layers], sep="\n")
        # print(i, error)
        # ax.relim()
        # ax.autoscale_view()
        return line, title

    if gif_params is not None:
        print("training the network and saving animation")
        with Timer(callback=Timer.DEFAULT_CALLBACK):
            anim = FuncAnimation(
                fig,
                update,
                init_func=lambda: (line, title),
                frames=iters,
                interval=gif_params.interval,
                blit=True,
            )
            anim.save(gif_params.filename, dpi=100, writer="imagemagick")

        # try:
        #     optimize(gif_params.filename)
        # except FileNotFoundError:
        #     logging.warning("gifsicle not installed")
    else:
        print("training the network")
        with Timer(callback=Timer.DEFAULT_CALLBACK):
            for i in range(iters):
                update(i)
        ax.plot(
            data.testing_input, data.testing_expected,
        )
        fig.show()

    return errors


def check_function(
    f: Callable[[np.ndarray], np.ndarray],
    input_data: Tuple[np.ndarray, np.ndarray],
    nn_params: NeuralNetworkParams,
    gif_params: Optional[GifParams],
) -> List[float]:
    training_input, testing_input = input_data
    expected = f(training_input)

    x_scaler = Scaler()
    x_scaler.fit(training_input)

    if isinstance(nn_params.layers[-1][1], Tanh):
        y_bounds = (-1, 1)
    else:
        y_bounds = (0, 1)

    y_scaler = Scaler(y_bounds)
    y_scaler.fit(expected)

    testing_expected = f(testing_input)

    errors = train_progress(
        Data(
            (training_input, expected),
            (testing_input, testing_expected),
            (x_scaler, y_scaler),
        ),
        nn_params,
        gif_params,
    )

    return errors


def main_square(test_name: str, make_gif: bool):
    return main_any(
        np.square,
        (
            np.linspace(-50, 50, 26).reshape(-1, 1),
            np.linspace(-50, 50, 101).reshape(-1, 1),
        ),
        test_name,
        make_gif,
    )


def main_sin(test_name: str, make_gif: bool):
    return main_any(
        lambda x: np.sin((3 * np.pi / 2) * x),
        (np.linspace(0, 2, 21).reshape(-1, 1), np.linspace(0, 2, 161).reshape(-1, 1)),
        test_name,
        make_gif,
    )


def main_any(
    f: Callable[[np.ndarray], np.ndarray],
    data: Tuple[np.ndarray, np.ndarray],
    test_name: str,
    make_gif: bool,
):
    test_id = 3
    if make_gif:
        dir_name = f"plots/{test_id}"
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        filename_template = f"{dir_name}/{test_name}-plot-{{len}}-{{activations}}.gif"

        def get_gif_params(activations):
            filename = filename_template.format(
                len=len(activations), activations="-".join(map(str, activations)),
            )
            return GifParams(filename, 50)

    else:
        # no gif will be created
        def get_gif_params(_):
            return None

    def perms(num: int):
        return permutations_with_replacement([Relu(), Sigmoid(), Tanh()], num)

    import csv

    with open(f"{test_name}-learning-{test_id}.csv", 'w+', newline='') as info_file:
        writer = csv.writer(info_file)
        for activations in chain(perms(2), perms(3)):
            gif_params = get_gif_params(activations)
            neuron_counts = [10] * len(activations)
            neuron_counts[-1] = 1
            with Timer() as timer:
                errors = check_function(
                    f,
                    data,
                    NeuralNetworkParams(list(zip(neuron_counts, activations)), (2000, 200)),
                    gif_params,
                )
            writer.writerow(["-".join(map(str, activations)), timer.elapsed_ns / 1e6, *errors])


def main(make_gifs=False):
    with Timer(callback=Timer.DEFAULT_CALLBACK):
        square_filename = "square"
        main_square(square_filename, make_gifs)
        sin_filename = "sin"
        main_sin(sin_filename, make_gifs)


if __name__ == "__main__":
    main()
