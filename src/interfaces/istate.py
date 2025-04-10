from abc import ABCMeta, abstractmethod


class IState:
    __metaclass__ = ABCMeta
    @abstractmethod
    def __str__(self) -> str:
        pass
    @property
    @abstractmethod
    def name(self):
        pass