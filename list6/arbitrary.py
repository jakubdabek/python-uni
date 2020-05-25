from typing import Tuple, Callable, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn import preprocessing

from list6.activation import Sigmoid, Tanh, Relu
from list6.neural_network import NeuralNetwork


def calc_error(expected, real) -> float:
    return np.square(expected - real).mean()


Scaler = preprocessing.MinMaxScaler


def train_progress(
    training_data: Tuple[np.ndarray, np.ndarray],
    testing_data: Tuple[np.ndarray, np.ndarray],
    scalers: Tuple[Scaler, Scaler],
    epochs: Tuple[int, int],
    gif_filename: Optional[str],
    gif_interval: int,
):
    training_input, training_expected = training_data
    testing_input, testing_expected = testing_data

    x_scaler, y_scaler = scalers

    training_input_scaled = x_scaler.transform(training_input)
    training_expected_scaled = y_scaler.transform(training_expected)
    testing_input_scaled = x_scaler.transform(testing_input)
    testing_expected_scaled = y_scaler.transform(testing_expected)

    single_epoch, iters = epochs

    nn = NeuralNetwork.new_random(1, [10, 1], [Tanh(), Tanh()])

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    ax.plot(
        testing_input, testing_expected,
    )

    (line,) = ax.plot(
        testing_input,
        y_scaler.inverse_transform(nn.predict_trans(testing_input_scaled)),
        "o",
    )
    title: plt.Text = ax.set_title("0 iterations")
    line: plt.Line2D

    # scaled_expected = y_scaler.inverse_transform(expected)
    # ylim = np.min(scaled_expected), np.max(scaled_expected)
    # ax.set_ylim(*ylim)

    fig.show()

    def update(i):
        i += 1
        nn.train_trans(training_input_scaled, training_expected_scaled, single_epoch, 1)
        output = y_scaler.inverse_transform(nn.predict_trans(testing_input_scaled))
        line.set_ydata(output)
        error = calc_error(testing_expected, output)
        title.set_text(f"{i * single_epoch} iterations (error: {error:.5})")
        # print(i, *[layer.input_weights for layer in nn.layers], sep="\n")
        print(i, error)
        # ax.relim()
        # ax.autoscale_view()
        return line, title

    if gif_filename is not None:
        anim = FuncAnimation(
            fig,
            update,
            init_func=lambda: (line, title),
            frames=iters,
            interval=gif_interval,
            blit=True,
        )
        anim.save(gif_filename, dpi=100, writer="imagemagick")
    else:
        for i in range(iters):
            update(i)
    ax.plot(
        testing_input, testing_expected, scalex=False, scaley=False,
    )
    fig.show()


def check_function(
    f: Callable[[np.ndarray], np.ndarray],
    training_input: np.ndarray,
    testing_input: np.ndarray,
    epochs: Tuple[int, int],
    gif_filename: str = None,
    gif_interval: int = 40,
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
        (training_input, expected),
        (testing_input, testing_expected),
        (x_scaler, y_scaler),
        epochs,
        gif_filename=gif_filename,
        gif_interval=gif_interval,
    )
    return training_input, expected


def main_square(gif_filename=None):
    return check_function(
        np.square,
        np.linspace(-50, 50, 26).reshape(-1, 1),
        np.linspace(-50, 50, 101).reshape(-1, 1),
        (2000, 200),
        gif_filename,
        100,
    )


def main_sin(gif_filename=None):
    return check_function(
        lambda x: np.sin((3 * np.pi / 2) * x),
        np.linspace(0, 2, 21).reshape(-1, 1),
        np.linspace(0, 2, 161).reshape(-1, 1),
        (200, 400),
        gif_filename,
    )


def main():
    _square_filename = "plot-square-1.gif"
    main_square(None)
    _sin_filename = "plot-sin-1.gif"
    main_sin(None)


if __name__ == "__main__":
    main()
