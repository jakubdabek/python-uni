import time
from contextlib import contextmanager
from functools import wraps


def print_elapsed_ns(elapsed_ns):
    print(f"Elapsed: {elapsed_ns / 1e6}ms")


def print_iter_elapsed_ns(elapsed_ns, iters):
    elapsed_ms = elapsed_ns / 1e6
    print(f"{iters:8} iters in {elapsed_ms}ms, avg: {elapsed_ms / iters}ms")


class Timer:
    __slots__ = ("_callback", "_start_time", "elapsed_ns")
    DEFAULT_CALLBACK = print_elapsed_ns
    DEFAULT_ITER_CALLBACK = print_iter_elapsed_ns

    def __init__(self, *, callback=None):
        self._callback = callback

    def check_elapsed_ns(self):
        self.elapsed_ns = elapsed_ns = time.perf_counter_ns() - self._start_time
        return elapsed_ns

    def __enter__(self):
        self._start_time = time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        elapsed_ns = self.check_elapsed_ns()
        if self._callback:
            self._callback(elapsed_ns)
        return False

    @classmethod
    def make_iter_timer(cls, period=100, *, intermediate_callback=None, callback=None):
        timer = cls(callback=callback)
        iters = 1

        if intermediate_callback is None:
            intermediate_callback = Timer.DEFAULT_ITER_CALLBACK

        def iter_completed():
            nonlocal iters
            elapsed_ns = timer.check_elapsed_ns()
            if iters % period == 0:
                intermediate_callback(elapsed_ns, iters)
            iters += 1

        @contextmanager
        def manager():
            try:
                yield
            finally:
                iter_completed()

        iter_completed.manager = manager

        return timer, iter_completed


def timed(wrapped):
    @wraps(wrapped)
    def wrapper(*args, **kwargs):
        with Timer(callback=print_elapsed_ns):
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
