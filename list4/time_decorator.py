import time
from functools import wraps


def timed(wrapped):
    @wraps(wrapped)
    def wrapper(*args, **kwargs):
        start = time.perf_counter_ns()
        result = wrapped(*args, **kwargs)
        end = time.perf_counter_ns()
        print(f"Elapsed: {(end - start) / 1e6}ms")

        return result

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
