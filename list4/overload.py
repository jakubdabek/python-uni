import inspect
import math
import operator
from contextlib import contextmanager
from typing import Callable, List, Tuple, Type
from collections import defaultdict


class NoApplicableOverloadError(TypeError):
    default_message = "No overload matching arguments has been registered"

    def __init__(self, *args):
        if not args:
            args = (NoApplicableOverloadError.default_message,)
        super().__init__(*args)


class AmbiguousOverloadError(Exception):
    default_message = "More than one applicable overload exists, " \
                      "and no one is more specific than the others"

    def __init__(self, *args):
        if not args:
            args = (AmbiguousOverloadError.default_message,)
        super().__init__(*args)


def _sig_param_type(param: inspect.Parameter) -> Type:
    cls = param.annotation
    if cls is inspect.Signature.empty:
        return object
    else:
        return cls


@contextmanager
def assert_raises(exc_cls=Exception, exact=False):
    try:
        yield
    except Exception as e:
        check = issubclass if not exact else operator.is_
        assert check(type(e), exc_cls), f"Expected {exc_cls.__name__}, got {type(e).__name__}"
    else:
        raise AssertionError(f"{exc_cls.__name__} not raised")


def main_assert_raises():
    with assert_raises(AssertionError):
        with assert_raises(TypeError):
            raise RuntimeError

    with assert_raises(Exception):
        raise RuntimeError

    with assert_raises(AssertionError):
        with assert_raises(Exception, exact=True):
            raise RuntimeError


# TODO: method overloading
class Overload:
    _instances = defaultdict(lambda: Overload())

    def __init__(self) -> None:
        self.functions: List[Tuple[inspect.Signature, Callable]] = []

    def register(self, func: Callable) -> None:
        self.functions.append((inspect.signature(func), func))

    def __call__(self, *args, **kwargs):
        best_candidate = None

        def better_candidate(current, new):
            if current is None:
                return new

            c_sig, c_f, c_bound = current
            n_sig, n_f, n_bound = new

            new_better = None

            def update_new_better(current_new_better, new_new_better):
                if current_new_better is None or current_new_better is new_new_better:
                    return new_new_better
                else:
                    raise AmbiguousOverloadError

            params = ((c_sig.parameters[name], n_sig.parameters[name]) for name in c_bound.arguments.keys())
            for c_cls, n_cls in ((_sig_param_type(c_param), _sig_param_type(n_param)) for c_param, n_param in params):
                if c_cls is n_cls:
                    continue

                if issubclass(n_cls, c_cls):
                    new_better = update_new_better(new_better, True)
                elif issubclass(c_cls, n_cls):
                    new_better = update_new_better(new_better, False)
                else:
                    raise AmbiguousOverloadError

            if new_better is True:
                return new
            else:
                return current

        for sig, f in self.functions:
            try:
                bound = sig.bind(*args, **kwargs)
            except TypeError:
                continue

            for name, value in bound.arguments.items():
                if not isinstance(value, _sig_param_type(sig.parameters[name])):
                    break
            else:
                best_candidate = better_candidate(best_candidate, (sig, f, bound))

        if best_candidate is not None:
            _, f, bound = best_candidate
            return f(*bound.args, **bound.kwargs)
        else:
            raise NoApplicableOverloadError

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
def check_type(x):
    return object


@overload
def check_type(x: int):
    return int


@overload
def check_type(x: str):
    return str


@overload
def ambiguous_simple(x, y: str):
    return object, str


@overload
def ambiguous_simple(x: str, y):
    return str, object


@overload
def kwd_only(x, y):
    return object, object


@overload
def kwd_only(x, *, y: int):
    return object, '*', int


@overload
def kwd_only2(x, *, y):
    return object, '*', object


@overload
def kwd_only2(x, y: int):
    return object, int


@overload
def defaults(x: int, y: str = 'def(y)'):
    return int, y


@overload
def defaults(x: str):
    return str


@overload
def order(x: str = 'def(x)', *, y='def(y)'):
    return x, y


@overload
def order(y: str = 'def(y)', *, x='def(x)'):
    return y, x


def foo(a, b: int, c=1, *args, x, y=None):
    pass


def main():
    print(f"norm(2,4)   = {norm(2, 4)}")
    print(f"norm(2,3,4) = {norm(2, 3, 4)}")
    print()

    print(f"norm2(2,4)   = {norm2(2, 4)}")
    print(f"norm2(2,3,4) = {norm2(2, 3, 4)}")
    print()

    print(f"check_type(1)   = {check_type(1)}")
    print(f"check_type('a') = {check_type('a')}")
    print(f"check_type([])  = {check_type([])}")
    print()

    print(f"ambiguous_simple('x', 1) = {ambiguous_simple('x', 1)}")
    print(f"ambiguous_simple(2, 'y') = {ambiguous_simple(2, 'y')}")
    print()

    with assert_raises(AmbiguousOverloadError):
        ambiguous_simple('x', 'y')

    with assert_raises(NoApplicableOverloadError):
        ambiguous_simple(1, 2)

    print(f"kwd_only('x', 'y')   = {kwd_only('x', 'y')}")
    print(f"kwd_only('x', y='y') = {kwd_only('x', y='y')}")
    print(f"kwd_only('x', y=1)   = {kwd_only('x', y=1)}")
    print()

    print(f"kwd_only2('x', 1)     = {kwd_only2('x', 1)}")
    print(f"kwd_only2('x', y='y') = {kwd_only2('x', y='y')}")
    print(f"kwd_only2('x', y=1)   = {kwd_only2('x', y=1)}")
    print()

    with assert_raises(NoApplicableOverloadError):
        kwd_only2('x', 'y')

    print(f"defaults(1, 'y') = {defaults(1, 'y')}")
    print(f"defaults(1)      = {defaults(1)}")
    print(f"defaults('x')    = {defaults('x')}")
    print()

    with assert_raises(NoApplicableOverloadError):
        defaults('x', 'y')

    with assert_raises(NoApplicableOverloadError):
        print(f"defaults(y='y')  = {defaults(y='y')}")

    print(f"order('x', y='y') = {order('x', y='y')}")
    print(f"order('y', x='x') = {order('y', x='x')}")
    print(f"order('x')        = {order('x')}")
    print(f"order(x='x')      = {order(x='x')}")
    print(f"order(y='y')      = {order(y='y')}")
    print(f"order(x=1)        = {order(x=1)}")
    print(f"order()           = {order()}")
    print()

    with assert_raises(AmbiguousOverloadError):
        order(x='x', y='y')

    with assert_raises(NoApplicableOverloadError):
        order('x', 'y')

    with assert_raises(NoApplicableOverloadError):
        order(1)


if __name__ == '__main__':
    main_assert_raises()
    main()
