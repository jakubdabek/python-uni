import itertools
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence, MutableSequence
from typing import Iterable


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

    def visit_dfs(self, visit):
        """Calls visit on every node in tree by doing a pre-order traversal."""
        if self.is_empty(self):
            return

        visit(self.value)
        for child in self.children():
            child.visit_dfs(visit)

    def visit_bfs(self, visit):
        """Calls visit on every node in tree by doing a level-order traversal."""
        if self.is_empty(self):
            return

        q = deque()
        q.append(self)
        while q:
            current: Tree = q.popleft()
            visit(current.value)
            q.extend(child for child in current.children() if not self.is_empty(child))

    @classmethod
    def random(cls, value_gen, depth):
        pass


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


Node = ListTree


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

    dfs, bfs = [], []
    tree.visit_dfs(dfs.append)
    tree.visit_bfs(bfs.append)

    print(f"dfs: {dfs}")
    print(f"bfs: {bfs}")

    assert dfs == ['1', '2', '4', '8', '9', '5', '3', '6', '7']
    assert bfs == list(map(str, range(1, 10)))


if __name__ == '__main__':
    main()
