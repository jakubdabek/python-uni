from typing import TypeVar, List

T = TypeVar("T")


def permutations_with_replacement(lst: List[T], size: int) -> List[List[T]]:
    if size < 1:
        return []
    if size == 1:
        return [[x] for x in lst]
    prev = permutations_with_replacement(lst, size - 1)
    return [[x] + rest for x in lst for rest in prev]