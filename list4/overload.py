import inspect
import math
from typing import Callable, List, Tuple, Type
from collections import defaultdict

from decorator import decorator


class NoApplicableOverloadError(TypeError):
    pass


def _sig_param_type(param: inspect.Parameter) -> Type:
    cls = param.annotation
    if cls is inspect.Signature.empty:
        return object
    else:
        return cls


class Overload:
    _instances = defaultdict(lambda: Overload())

    def __init__(self) -> None:
        self.functions: List[Tuple[inspect.Signature, Callable]] = []

    def register(self, func: Callable) -> None:
        self.functions.append((inspect.signature(func), func))

    def __call__(self, *args, **kwargs):
        for sig, f in self.functions:
            try:
                bound = sig.bind(*args, **kwargs)
            except TypeError:
                continue

            for name, value in bound.arguments.items():
                if not isinstance(value, _sig_param_type(sig.parameters[name])):
                    break
            else:
                bound.apply_defaults()
                # TODO: e.g. list > object
                return f(*bound.args, **bound.kwargs)

        raise NoApplicableOverloadError("No overload matching arguments has been registered")

    @classmethod
    def from_function(cls, func):
        ov = cls._instances[func.__name__]
        ov.register(func)
        return ov


def overload(wrapped):
    return Overload.from_function(wrapped)


@overload
def norm(x, y):
    return math.sqrt(x * x + y * y)


@overload
def norm(x, y, z):
    return abs(x) + abs(y) + abs(z)


@overload
def norm2(x, y, z):
    return abs(x) + abs(y) + abs(z)


@overload
def norm2(x, y):
    return math.sqrt(x * x + y * y)


@overload
def check_type(x: int):
    return int


@overload
def check_type(x: str):
    return str


@overload
def check_type(x):
    return object


def foo(a, b: int, c=1, *args, x, y=None):
    pass


def main():
    print(f"norm(2,4)   = {norm(2, 4)}")
    print(f"norm(2,3,4) = {norm(2, 3, 4)}")

    print(f"norm2(2,4)   = {norm2(2, 4)}")
    print(f"norm2(2,3,4) = {norm2(2, 3, 4)}")

    print(f"typecheck(1)   = {check_type(1)}")
    print(f"typecheck('a') = {check_type('a')}")
    print(f"typecheck([])  = {check_type([])}")


if __name__ == '__main__':
    main()
