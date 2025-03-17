from abc import ABC, abstractmethod


class Writer(ABC):
    @abstractmethod
    def write(self, metric):
        pass


class NullWriter(Writer):
    def write(self, metric):
        pass


class ConsoleWriter(Writer):
    def write(self, metric):
        print(metric)
