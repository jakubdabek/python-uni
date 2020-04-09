from abc import ABC, abstractmethod


class Tree(ABC):
    @property
    @abstractmethod
    def left(self):
        raise NotImplementedError

    @left.setter
    @abstractmethod
    def left(self, val):
        raise NotImplementedError

    @property
    @abstractmethod
    def right(self):
        raise NotImplementedError

    @right.setter
    @abstractmethod
    def right(self, val):
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
    def with_value(cls, value):
        raise NotImplementedError

    @classmethod
    def random(cls, value_gen, depth):
        pass


class ListTree(Tree):
    __init_sentinel = object()

    def __init__(self, list_, *, sentinel=None):
        if sentinel is not ListTree.__init_sentinel:
            raise ValueError("Use empty or with_value constructors")
        if not (list_ is None or len(list_) == 3):
            raise TypeError("Argument should be an empty node (None) or a list with 3 elements")
        self._list = list_

    def __str__(self):
        return str(self._list)

    def __repr__(self):
        return f"ListTree({repr(self._list)})"

    @property
    def left(self):
        return self._list[1]

    @left.setter
    def left(self, val):
        if not isinstance(val, ListTree):
            raise TypeError
        self._list[1] = val

    @property
    def right(self):
        return self._list[2]

    @right.setter
    def right(self, val):
        if not isinstance(val, ListTree):
            raise TypeError
        self._list[2] = val

    @property
    def value(self):
        return self._list[0]

    @value.setter
    def value(self, val):
        self._list[0] = val

    @staticmethod
    def is_empty(tree):
        return tree is None

    @classmethod
    def empty(cls):
        return cls(None, sentinel=cls.__init_sentinel)

    @classmethod
    def with_value(cls, value):
        return cls([value, cls.empty(), cls.empty()], sentinel=cls.__init_sentinel)
