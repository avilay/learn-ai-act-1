from abc import ABC, abstractmethod
from .serializable import Serializable


class Writer(ABC):
    @abstractmethod
    def write(self, metric: Serializable):
        pass


class NullWriter(Writer):
    def write(self, metric: Serializable):
        pass


class ConsoleWriter(Writer):
    def write(self, metric: Serializable):
        if metric is not Serializable:
            print('metric is not Serializable!')
            return
        print(metric)
