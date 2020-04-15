import math
import random
from cmath import exp
from math import pi
from typing import List, TypeVar

import numpy as np


def calculate_omega(k, n):
    return exp(-2j * k * pi / n)


def calculate_omega2(k, n):
    return math.cos(2 * k * pi / n) + math.sin(2 * k * pi / n) * 1j


def dft(x, n):
    return [
        sum(
            x[i] * calculate_omega(i * k, n) if i < len(x) else 0
            for i in range(n)
        )
        for k in range(n)
    ]


def idft(x, n):
    return [
        int(round(sum(
            x[i] * calculate_omega(-i * k, n) if i < len(x) else 0
            for i in range(n)
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
    return [int(x * inv + 0.5) for x in arr[:n].real]


def null(*args, **kwargs):
    pass


def my_dft_rec(arr: np.ndarray, omega: complex, inv=False, *, debug=null, depth=0) -> None:
    if len(arr) <= 1:
        return

    debug(' ' * depth, f"in (omega={omega:.4f}): ", np.array2string(arr, max_line_width=200, precision=3))
    arr_even = arr[::2]
    arr_odd = arr[1::2]

    omega2 = omega ** 2
    my_dft_rec(arr_even, omega2, inv, debug=debug, depth=depth + 2)
    my_dft_rec(arr_odd, omega2, inv, debug=debug, depth=depth + 2)

    debug(' ' * depth, f"mid(omega={omega:.4f}): ", np.array2string(arr, max_line_width=200, precision=3))
    arr_half = np.empty(len(arr) // 2, dtype=np.complex_)

    if inv:
        omega = omega.conjugate()
    x = 1 + 0j
    for i, (even, odd) in enumerate(zip(arr_even, arr_odd)):
        debug(' ' * depth, even, odd, end=" ")
        arr[i] = even + x * odd
        arr_half[i] = even - x * odd
        x *= omega

    debug()
    arr[len(arr) // 2:] = arr_half

    debug(' ' * depth, f"out(omega={omega:.4f}): ", np.array2string(arr, max_line_width=200, precision=3))


def np_dft(x, n):
    return np.fft.fft(x, n)


def np_idft(x, n):
    return [int(x + 0.5) for x in np.fft.ifft(x, n).real]


T = TypeVar('T')


def rstrip(l: List[T], elem: T) -> None:
    while l[-1] == elem:
        l.pop()


class FastBigNum:
    fourier = (dft, idft)

    def __init__(self, digits: List[int]):
        rstrip(digits, 0)
        self.digits = digits

    @classmethod
    def from_str(cls, s: str):
        return cls(list(map(int, reversed(s))))

    @classmethod
    def from_int(cls, n: int):
        return cls.from_str(str(n))

    @classmethod
    def normalized(cls, lst: List[int]):
        carry = 0
        for i, val in enumerate(lst):
            carry, lst[i] = divmod(val + carry, 10)

        return cls(lst)

    def __mul__(self, other):
        if isinstance(other, int):
            return self * type(self).from_int(other)
        elif isinstance(other, str):
            return self * type(self).from_str(other)
        elif type(self) is not type(other):
            return NotImplemented

        n = max(len(self.digits), len(other.digits))

        local_dft, local_idft = self.fourier

        x_star = local_dft(self.digits, 2 * n)
        y_star = local_dft(other.digits, 2 * n)
        z_star = [x * y for x, y in zip(x_star, y_star)]
        z = local_idft(z_star, 2 * n)

        return type(self).normalized(z)

    def __int__(self):
        return int(str(self))

    def __str__(self):
        return ''.join(map(str, reversed(self.digits)))

    def __repr__(self):
        return f"FastBigNum({str(self)})"


def test(fourier, a: int, b: int):
    FastBigNum.fourier = fourier

    fast_a = FastBigNum.from_int(a)
    fast_b = FastBigNum.from_int(b)
    fast_result = fast_a * fast_b

    real_result = a * b

    print(f"results for {fourier}:")
    print("fft:      ", fast_result)
    print("standard: ", real_result)
    if int(fast_result) == real_result:
        print("ok, same")
    else:
        print("!wrong!")

    # assert int(fast_result) == real_result, f"wrong result: {fast_result} == {real_result}"


def main():
    for fourier in [(dft, idft), (np_dft, np_idft), (my_dft, my_idft)]:
        for a, b in [
            (135, 210),
            (13123122312321312312312231231231231212331233231349, 12123123112231231213123312321321231231112323123231)
        ]:
            test(fourier, a, b)
            print()

        for _ in range(10):
            a = ''.join([random.choice("0123456789") for _ in range(500)])
            b = ''.join([random.choice("0123456789") for _ in range(500)])
            test(fourier, int(a), int(b))
            print()


if __name__ == '__main__':
    main()
