import time
from functools import wraps
from contextlib import AbstractContextManager


class Timer(AbstractContextManager):
    def __init__(self, *, callback=None):
        self._callback = callback

    def __enter__(self):
        self._start_time = time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter_ns() - self._start_time
        if self._callback:
            self._callback(self.elapsed)
        return False


def print_elapsed(elapsed):
    print(f"Elapsed: {elapsed / 1e6}ms")


def timed(wrapped):
    @wraps(wrapped)
    def wrapper(*args, **kwargs):
        with Timer(callback=print_elapsed):
            return wrapped(*args, **kwargs)

    return wrapper


def foo(a, b, c=5, *, x):
    print(f"in foo({a}, {b}, {c}, *, {x})")
    time.sleep(1.234567)
    print(f"end foo({a}, {b}, {c}, *, {x})")

    return c * x


@timed
def foo_timed(a, b, c=5, *, x):
    return foo(a, b, c, x=x)


def main():
    print(foo(1, "a", x="x"))
    print(foo_timed(2, "b", x="y"))


if __name__ == '__main__':
    main()
