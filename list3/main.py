import random
from collections import abc
from typing import List, Iterator, Any, Iterable


def transpose(matrix: List[str]) -> List[str]:
    """Transposes a list in a format ['x11 x12 ...', 'x21 x22 ...', ...]"""
    return [' '.join(row) for row in zip(*(s.split() for s in matrix))]


def main_transpose():
    assert transpose(["1.1 2.2 3.3", "4.4 5.5 6.6", "7.7 8.8 9.9"]) \
           == \
           ["1.1 4.4 7.7", "2.2 5.5 8.8", "3.3 6.6 9.9"]

    assert transpose(["1.1 2.2", "4.4 5.5", "7.7 8.8"]) \
           == \
           ["1.1 4.4 7.7", "2.2 5.5 8.8"]

    assert transpose(["1.1 2.2 3.3", "4.4 5.5 6.6"]) \
           == \
           ["1.1 4.4", "2.2 5.5", "3.3 6.6"]

    assert transpose(["1.1 2.2 3.3"]) \
           == \
           ["1.1", "2.2", "3.3"]


def is_iterable(a: Any) -> bool:
    return isinstance(a, abc.Iterable) and not (isinstance(a, str) and len(a) <= 1)


def flatten(nested: Any) -> Iterator[Any]:
    if is_iterable(nested):
        for elem in nested:
            yield from flatten(elem)
    else:
        yield nested


def main_flatten():
    assert list(flatten([[1, 2, ["a", 4, "b", 5, 5, b'abc', 5, "123"]], [4, 5, 6], 7, '', [[9, [123, [[123]]]], 10]])) \
           == \
           [1, 2, 'a', 4, 'b', 5, 5, ord('a'), ord('b'), ord('c'), 5, '1', '2', '3', 4, 5, 6, 7, '', 9, 123, 123, 10]


def sum_last_field(lines: Iterable[str]) -> int:
    return sum(int(line.split()[-1]) for line in lines)


def main_sum_last_field():
    lines = """  127.0.0.1 -  - "GET /ply/  HTTP/1.1" 200 7587
  127.0.0.1 -  - "GET /favicon.ico HTTP/1.1" 404 133
  127.0.0.1 -  - "GET /ply/bookplug.gif" 200 23903
  127.0.0.1 -  - "GET /ply/ply.html HTTP/1.1" 200 97238
  127.0.0.1 -  - "GET /ply/example.html HTTP/1.1" 200 2359
  127.0.0.1 -  - "GET /index.html" 200 4447
""".splitlines()
    assert sum_last_field(lines) == 135667


def quicksort(a: List[int]) -> List[int]:
    """ qsort [] = []
        qsort (x:xs) = qsort elts_lt_x ++ [x] ++ qsort elts_greq_x
                       where
                           elts_lt_x = [y | y <- xs, y < x]
                           elts_greq_x = [y | y <- xs, y >= x]"""
    if not a:
        return a
    pivot = a[0]
    less = [x for x in a[1:] if x < pivot]
    greater_equal = [x for x in a[1:] if x >= pivot]
    return quicksort(less) + [pivot] + quicksort(greater_equal)


def quicksort_main():
    for _ in range(100):
        a = random.choices(range(2 ** 20), k=random.randint(0, 100))
        assert quicksort(a) == sorted(a)


def all_subsets(a: List[Any]):
    """let rec allsubsets s =
           match s with
               [] -> [[]]
               | (a::t) -> let res = allsubsets t in
                               map (fun b -> a::b) res @ res;;

    # allsubsets [1;2;3];;
    - : int list list = [[1; 2; 3]; [1; 2]; [1; 3]; [1]; [2; 3]; [2]; [3]; []]"""
    if not a:
        return [a]
    first = a[0]
    rest = all_subsets(a[1:])
    return [([first] + tail) for tail in rest] + rest


def all_subsets_main():
    assert all_subsets([1, 2, 3]) == [[1, 2, 3], [1, 2], [1, 3], [1], [2, 3], [2], [3], []]


def main():
    main_transpose()
    main_flatten()
    main_sum_last_field()
    quicksort_main()
    all_subsets_main()


if __name__ == '__main__':
    main()
