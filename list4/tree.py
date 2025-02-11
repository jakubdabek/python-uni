import itertools
import random
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence, MutableSequence
from typing import Iterable, Union


class Tree(ABC, Sequence):
    @abstractmethod
    def children(self) -> Iterable['Tree']:
        raise NotImplementedError

    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError

    @value.setter
    @abstractmethod
    def value(self, val):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def is_empty(tree):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def empty(cls):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def with_value(cls, value, *children):
        raise NotImplementedError

    def visit_dfs(self):
        """Calls visit on every node in tree by doing a pre-order traversal."""
        if self.is_empty(self):
            return

        yield self.value
        for child in self.children():
            yield from child.visit_dfs()

    BFS_LAST_IN_ROW = object()

    def visit_bfs(self, *, signal_last_in_row=False):
        """Calls visit on every node in tree by doing a level-order traversal."""
        if self.is_empty(self):
            return

        q = deque()
        q.append(self)
        if signal_last_in_row:
            q.append(Tree.BFS_LAST_IN_ROW)

        while q:
            current: Union[Tree, Tree.BFS_LAST_IN_ROW] = q.popleft()
            if current is Tree.BFS_LAST_IN_ROW:
                yield current
                if q:
                    q.append(Tree.BFS_LAST_IN_ROW)
            else:
                yield current.value
                q.extend(child for child in current.children() if not self.is_empty(child))

    @classmethod
    def random(cls, value_gen, child_num_gen, depth) -> 'Tree':
        if depth == 0:
            return cls.empty()

        child_num = child_num_gen()
        deepest_index = random.randrange(0, child_num)

        def next_depth(i):
            if i == deepest_index:
                return depth - 1
            else:
                return random.randint(0, depth - 1)

        children = (cls.random(value_gen, child_num_gen, next_depth(i)) for i in range(child_num))
        return cls.with_value(value_gen(), *children)


class ListTree(Tree, MutableSequence):
    __init_sentinel = object()

    def __init__(self, list_, *, sentinel=None):
        if sentinel is not ListTree.__init_sentinel:
            raise ValueError("Use empty or with_value constructors")
        self._list = list_

    def __str__(self):
        return str(self._list)

    def __repr__(self):
        return f"ListTree({repr(self._list)})"

    def __getitem__(self, i: int) -> 'ListTree':
        self._assert_nonempty()
        if i >= 0:
            i = i + 1
        elif -i <= len(self._list) - 1:
            pass
        else:
            raise IndexError("list index out of range")

        return self._list.__getitem__(i)

    def __setitem__(self, i: int, child: 'ListTree') -> None:
        self._assert_nonempty()
        if not isinstance(child, ListTree):
            raise TypeError
        return self._list.__setitem__(i + 1, child)

    def __delitem__(self, i: int) -> None:
        self._assert_nonempty()
        return self._list.__delitem__(i + 1)

    def __len__(self) -> int:
        if self.is_empty(self):
            return 0
        return self._list.__len__() - 1

    def children(self) -> Iterable['ListTree']:
        return itertools.islice(self._list or (), 1, None)

    def insert(self, index: int, child: 'ListTree') -> None:
        self._assert_nonempty()
        if not isinstance(child, ListTree):
            raise TypeError
        return self._list.insert(index + 1, child)

    @property
    def value(self):
        return self._list[0]

    @value.setter
    def value(self, val):
        if self.is_empty(self):
            self._list = [val]
            return
        self._list[0] = val

    def _assert_nonempty(self):
        if self.is_empty(self):
            raise ValueError("Unable to perform operation on an empty tree, set the value first")

    @staticmethod
    def is_empty(tree):
        return isinstance(tree, ListTree) and tree._list is None

    @classmethod
    def empty(cls):
        return cls(None, sentinel=cls.__init_sentinel)

    @classmethod
    def with_value(cls, value, *children):
        return cls([value, *children], sentinel=cls.__init_sentinel)

    @classmethod
    def from_list(cls, list_):
        if list_ is None:
            return cls.empty()
        return cls.with_value(list_[0], *map(cls.from_list, itertools.islice(list_, 1, None)))


class Node(ListTree):
    def __repr__(self):
        if self.is_empty(self):
            return "Node(EMPTY)"
        return f"Node(value={self.value}, children=[{', '.join(repr(c) for c in self.children())}])"

    def __str__(self):
        return self.__repr__()


def main():
    tree = ListTree.from_list(
        ["1",
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
    )

    dfs = list(tree.visit_dfs())
    bfs = list(tree.visit_bfs())

    print(f"dfs: {dfs}")
    print(f"bfs: {bfs}")

    assert dfs == ['1', '2', '4', '8', '9', '5', '3', '6', '7']
    assert bfs == list(map(str, range(1, 10)))

    def print_child(current):
        if current is Tree.BFS_LAST_IN_ROW:
            print()
        else:
            print(current, end=" ")

    def print_bfs(t):
        for c in t.visit_bfs(signal_last_in_row=True):
            print_child(c)

    print_bfs(tree)
    print()

    random_tree = ListTree.random(lambda: random.randint(1, 100), lambda: random.randint(2, 4), 5)
    print_bfs(random_tree)
    print()

    random_node_tree = Node.random(lambda: random.randint(1, 100), lambda: random.randint(2, 4), 5)
    print_bfs(random_node_tree)
    print(random_node_tree)


if __name__ == '__main__':
    main()
