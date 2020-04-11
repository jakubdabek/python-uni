import itertools
import random
from abc import ABC
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Iterable, Optional, NoReturn, Union, Type, Callable

from list4.overload import overload


class TreeABC(ABC):
    pass


@dataclass
class Node:
    value: Any
    children: List[Optional['Node']]

    @classmethod
    def from_lazy(cls, v: Any, c: Iterable[Optional['Node']]):
        return cls(v, list(c))


ListTree = Optional[List]
# TreeABC.register(ListTree)

TreeABC.register(list)
TreeABC.register(type(None))


@overload
def value(tree: Node) -> Any:
    return tree.value


@overload
def value(tree: List) -> Any:
    return tree[0]


@overload
def value(tree: type(None)) -> NoReturn:
    raise ValueError("Empty tree has no value")


@overload
def children(tree: Node) -> Iterable[Node]:
    return tree.children


@overload
def children(tree: List) -> Iterable[ListTree]:
    return itertools.islice(tree, 1, None)


@overload
def children(tree: type(None)) -> NoReturn:
    raise ValueError("Empty tree has no children")


@overload
def is_tree_empty(tree: List) -> bool:
    return False


@overload
def is_tree_empty(tree: type(None)) -> bool:
    return True


@overload
def is_tree_empty(tree: Node) -> bool:
    return False


def make_empty_list_tree() -> ListTree:
    return None


def make_list_tree(value: Any, children: Iterable[ListTree]) -> ListTree:
    return [value, *children]


def _next_depth(i, deepest_index, current_height):
    if i == deepest_index:
        return current_height - 1
    else:
        return random.randint(0, current_height - 1)


def random_tree(
        make_empty: Callable[[], TreeABC],
        make_full: Callable[[Any, Iterable[TreeABC]], TreeABC],
        value_gen: Callable[[], Any],
        child_num_gen: Callable[[], int],
        height: int
) -> TreeABC:
    def impl(current_height):
        if current_height == 0:
            return make_empty()

        child_num = child_num_gen()
        deepest_index = random.randrange(0, child_num)

        children = (impl(_next_depth(i, deepest_index, current_height)) for i in range(child_num))
        return make_full(value_gen(), children)

    return impl(height)


def traverse_dfs(tree: TreeABC) -> Iterable[Any]:
    if is_tree_empty(tree):
        return

    yield value(tree)
    for child in children(tree):
        yield from traverse_dfs(child)


BFS_LAST_IN_ROW = object()


def traverse_bfs(tree: TreeABC, *, signal_last_in_row=False) -> Iterable[Any]:
    if is_tree_empty(tree):
        return

    q = deque()
    q.append(tree)
    if signal_last_in_row:
        q.append(BFS_LAST_IN_ROW)

    while q:
        current: Union[TreeABC, BFS_LAST_IN_ROW] = q.popleft()
        if current is BFS_LAST_IN_ROW:
            yield current
            if q:
                q.append(BFS_LAST_IN_ROW)
        else:
            yield value(current)
            q.extend(child for child in children(current) if not is_tree_empty(child))


def main():
    values = ["1",
              ["2",
               ["4",
                ["8", None, None],
                ["9", None, None]
                ],
               ["5", None, None],
               ],
              ["3",
               ["6", None, None],
               ["7", None, None],
               ]
              ]

    def print_child(current):
        if current is BFS_LAST_IN_ROW:
            print()
        else:
            print(current, end=" ")

    def print_bfs(t: TreeABC) -> None:
        for c in traverse_bfs(t, signal_last_in_row=True):
            print_child(c)

    def print_all(t: TreeABC) -> None:
        print_bfs(t)
        print(list(traverse_dfs(t)))
        print(list(traverse_bfs(t)))
        print(t)
        print()

    print_all(values)

    tree1 = random_tree(
        make_empty_list_tree,
        make_list_tree,
        lambda: random.randint(1, 100),
        lambda: random.randint(2, 4),
        4
    )
    print_all(tree1)

    tree2 = random_tree(
        type(None),
        Node.from_lazy,
        lambda: random.randint(1, 100),
        lambda: random.randint(2, 4),
        4
    )
    print_all(tree2)


if __name__ == '__main__':
    main()
