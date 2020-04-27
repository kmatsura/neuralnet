from abc import ABCMeta, abstractmethod
from typing import List, Callable


class AbstractNode(metaclass=ABCMeta):
    @abstractmethod
    def process_input(self, *args: List[float]) -> float:
        """
        Integrate input signals.
        *args is the tupple of input value from the previous layer.
        """
        raise NotImplementedError

    @abstractmethod
    def activate(self, activation_function: Callable[[float], float],
                 inputsignal: float) -> float:
        """
        Map input args to output value.
        """
        raise NotImplementedError


class SimpleNode(AbstractNode):
    def __init__(self):
        super().__init__()

    def process_input(self, *args: List[float]) -> float:
        """
        Summing up input signals.
        """
        value = sum(*args)
        return value

    def activate(self, activation_function: Callable[[float], float],
                 inputsignal: float) -> float:
        output = activation_function(inputsignal)
        return output
