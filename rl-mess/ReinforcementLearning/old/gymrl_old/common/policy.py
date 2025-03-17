from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def action(self, state):
        pass

    @abstractmethod
    def pmf(self, state):
        pass

    @abstractmethod
    def prob(self, state, action):
        pass
