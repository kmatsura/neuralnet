from abc import ABCMeta, abstractmethod


class AbstractNode(metaclass=ABCMeta):
    @abstractmethod
    def input(self, *args):
        """
        Integrate input signals.
        *args is the tupple of input value from the previous layer.
        """
        raise NotImplementedError

    @abstractmethod
    def activate(self, parameter_list):
        """
        Map input args to output value.
        """
        raise NotImplementedError

    @abstractmethod
    def output(self, weight):
        raise NotImplementedError


class SimpleNode(AbstractNode):
    def __init__(self):
        super().__init__()

    def input(self, *args):
        """
        Summing up input signals.
        """
        value = sum(*args)
        return value

    def activate(self, parameter_list):
        return super().activate(parameter_list)
