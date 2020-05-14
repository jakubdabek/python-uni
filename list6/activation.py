from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

ActivationT = TypeVar("ActivationT", float, np.ndarray)


class Activation(ABC):
    @staticmethod
    @abstractmethod
    def value(x: ActivationT) -> ActivationT:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def gradient(x: ActivationT) -> ActivationT:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def gradient_from_value(x: ActivationT) -> ActivationT:
        raise NotImplementedError


class Sigmoid(Activation):
    @staticmethod
    def value(x: ActivationT) -> ActivationT:
        return 1.0 / (1.0 + np.exp(np.negative(x)))

    @staticmethod
    def gradient(x: ActivationT) -> ActivationT:
        fx = Sigmoid.value(x)
        return Sigmoid.gradient_from_value(fx)

    @staticmethod
    def gradient_from_value(x: ActivationT) -> ActivationT:
        return x * (1.0 - x)


class Relu(Activation):
    @staticmethod
    def value(x: ActivationT) -> ActivationT:
        return np.maximum(x, 0)

    @staticmethod
    def gradient(x: ActivationT) -> ActivationT:
        return np.where(x > 0, 1, 0)

    @staticmethod
    def gradient_from_value(x: ActivationT) -> ActivationT:
        return np.where(x > 0, 1, 0)


class Tanh(Activation):
    @staticmethod
    def value(x: ActivationT) -> ActivationT:
        return np.tanh(x)

    @staticmethod
    def gradient(x: ActivationT) -> ActivationT:
        fx = Tanh.value(x)
        return Tanh.gradient_from_value(fx)

    @staticmethod
    def gradient_from_value(x: ActivationT) -> ActivationT:
        return 1.0 - np.square(x)
