import math
import random
import sys
import timeit
from cmath import exp
from math import pi
from pathlib import Path
from typing import List, TypeVar, Collection, Union

import numpy as np

from list4 import time_decorator
from list4.overload import flatten
from list4.time_decorator import Timer


def calculate_omega(k, n):
    return exp(-2j * k * pi / n)


def calculate_omega2(k, n):
    return math.cos(2 * k * pi / n) + math.sin(2 * k * pi / n) * 1j


def dft(x, n):
    return [
        sum(
            elem * calculate_omega(i * k, n)
            for i, elem in enumerate(x)
        )
        for k in range(n)
    ]


def idft(x, n):
    return [
        int(round(sum(
            elem * calculate_omega(-i * k, n)
            for i, elem in enumerate(x)
        ).real) / n)
        for k in range(n)
    ]


def extend_to_pow(a: int, base: int = 2) -> int:
    a_extended = 1
    while a_extended < a:
        a_extended *= base

    return a_extended


# http://www.cs.rug.nl/~ando/pdfs/Ando_Emerencia_multiplying_huge_integers_using_fourier_transforms_paper.pdf
def my_dft(x, n):
    n_extended = extend_to_pow(n)
    x = x[:n]

    omega = calculate_omega2(1, n_extended)
    arr = np.empty(n_extended, np.complex_)
    arr[:len(x)] = x
    arr[len(x):] = 0

    my_dft_rec(arr, omega)

    return arr


def my_idft(arr, n):
    arr = np.array(arr)
    n_extended = extend_to_pow(n)
    omega = calculate_omega2(1, n_extended)

    my_dft_rec(arr, omega, inv=True)

    inv = 1 / n_extended
    # inv = 1
    return (arr[:n].real * inv + 0.5).astype(np.int_)


def debug_print(lambda_):
    lambda_(print)


def my_dft_rec(arr: np.ndarray, omega: complex, inv=False, *, debug=lambda _: _, depth=0) -> None:
    if len(arr) <= 1:
        return

    debug(lambda f: f(' ' * depth, f"in (omega={omega:.4f}): ", np.array2string(arr, max_line_width=200, precision=3)))
    arr_even = arr[::2]
    arr_odd = arr[1::2]

    omega2 = omega ** 2
    my_dft_rec(arr_even, omega2, inv, debug=debug, depth=depth + 2)
    my_dft_rec(arr_odd, omega2, inv, debug=debug, depth=depth + 2)

    debug(lambda f: f(' ' * depth, f"mid(omega={omega:.4f}): ", np.array2string(arr, max_line_width=200, precision=3)))
    arr_half = np.empty(len(arr) // 2, dtype=np.complex_)

    if inv:
        omega = omega.conjugate()
    x = 1 + 0j
    for i, (even, odd) in enumerate(zip(arr_even, arr_odd)):
        debug(lambda f: f(' ' * depth, even, odd, end=" "))
        arr[i] = even + x * odd
        arr_half[i] = even - x * odd
        x *= omega

    debug(lambda f: f())
    arr[len(arr) // 2:] = arr_half

    debug(lambda f: f(' ' * depth, f"out(omega={omega:.4f}): ", np.array2string(arr, max_line_width=200, precision=3)))


def np_dft(x, n):
    return np.fft.fft(x, n)


def np_idft(x, n):
    return np.floor(np.fft.ifft(x, n).real + 0.5).astype(np.int_)


T = TypeVar('T')


def rstrip(lst: Collection[T], pattern: T) -> Collection[T]:
    for i, elem in enumerate(reversed(lst)):
        if elem != pattern:
            return lst[:len(lst) - i]

    return lst


class FastBigNum:
    fourier = (dft, idft)

    def __init__(self, digits: Union[np.ndarray, List[int]]):
        if not isinstance(digits, np.ndarray):
            digits = np.array(digits)
        self.digits = rstrip(digits, 0)

    def normalize(self):
        if not isinstance(self.digits, np.ndarray):
            self.digits = np.array(self.digits)

        lst = self.digits

        carry = 0
        for i, val in enumerate(lst):
            if carry > 0 or val >= 10:
                carry, lst[i] = divmod(val + carry, 10)

        carried = []
        while carry > 0:
            carry, m = divmod(carry, 10)
            carried.append(m)

        if carried:
            self.digits = np.append(lst, carried)

    @classmethod
    def from_str(cls, s: str):
        return cls(np.fromiter(map(int, reversed(s)), dtype=np.int_))

    @classmethod
    def from_int(cls, n: int):
        return cls.from_str(str(n))

    @classmethod
    def normalized(cls, lst: List[int]):
        obj = cls(lst)
        obj.normalize()
        return obj

    mul_count = 0

    def __mul__(self, other):
        if isinstance(other, int):
            return self * type(self).from_int(other)
        elif isinstance(other, str):
            return self * type(self).from_str(other)
        elif type(self) is not type(other):
            return NotImplemented

        FastBigNum.mul_count += 1

        n = max(len(self.digits), len(other.digits))

        local_dft, local_idft = self.fourier

        x_star = local_dft(self.digits.copy(), 2 * n)
        y_star = local_dft(other.digits.copy(), 2 * n)

        if not isinstance(x_star, np.ndarray):
            x_star = np.array(x_star)
        if not isinstance(y_star, np.ndarray):
            y_star = np.array(y_star)

        z_star = x_star * y_star
        z = local_idft(z_star, 2 * n)

        return type(self)(z)

    def __int__(self):
        return int(str(self))

    def __str__(self):
        return ''.join(map(str, reversed(self.digits)))

    def __repr__(self):
        return f"FastBigNum({self.digits})"


def test(fourier, a: int, b: int, exc=False):
    FastBigNum.fourier = fourier

    fast_a = FastBigNum.from_int(a)
    fast_b = FastBigNum.from_int(b)
    fast_result = fast_a * fast_b * fast_a * fast_b

    real_result = a * b * a * b

    print(f"results for {fourier}:")
    print(f"{a} | {b}")
    print(f"{fast_a} | {fast_b}")
    print()
    print("fft:      ", repr(fast_result))
    fast_result.normalize()
    print("fft:      ", repr(fast_result))
    print("fft:      ", fast_result)
    print("standard: ", real_result)
    if int(fast_result) == real_result:
        print("ok, same")
    else:
        print("!wrong!")
        print("!!!!!!!", file=sys.stderr)
        if exc:
            raise RuntimeError("Wrong result")

    # assert int(fast_result) == real_result, f"wrong result: {fast_result} == {real_result}"


def main():
    for fourier in [(dft, idft), (np_dft, np_idft), (my_dft, my_idft)]:
        for a, b in [
            (135, 210),
            (13123122312321312312312231231231231212331233231349, 12123123112231231213123312321321231231112323123231)
        ]:
            test(fourier, a, b)
            print()

        size = 100
        for _ in range(10):
            a = ''.join(map(str, rstrip(np.random.choice(10, size), 0)))
            b = ''.join(map(str, rstrip(np.random.choice(10, size), 0)))
            test(fourier, int(a), int(b), exc=True)
            print()


def main_benchmark():
    lengths = list(flatten(((10**x, 5 * 10**x) for x in range(6))))
    print(lengths)

    for name, fourier in [
            ("fast", (my_dft, my_idft)),
            ("numpy", (np_dft, np_idft)),
            ("naive", (dft, idft)),
    ]:
        def setup(size):
            # with time_decorator.Timer(callback=Timer.DEFAULT_CALLBACK):
            a = np.random.choice(10, size)
            b = np.random.choice(10, size)

            return FastBigNum.normalized(a), FastBigNum.normalized(b)

        FastBigNum.fourier = fourier

        print(f"{name}:")
        results = []
        if name == "naive":
            local_lengths = lengths[:7]
        else:
            local_lengths = lengths

        for length in local_lengths:
            with time_decorator.Timer(callback=Timer.DEFAULT_CALLBACK):
                timer = timeit.Timer(
                    "a * b",
                    f"a, b = setup({length})",
                    globals=dict(**globals(), setup=setup)
                )
                number = 2
                result = np.mean(np.array(timer.repeat(repeat=3, number=number)) / number)
                print(f"{length:9}: {result}")
                results.append(result)

        print(f"muls: {FastBigNum.mul_count}")
        FastBigNum.mul_count = 0

        Path(f"./result_{name}.csv")\
            .write_text("\n".join(f"{length}, {result}" for length, result in zip(lengths, results)))


if __name__ == '__main__':
    # main()
    main_benchmark()
