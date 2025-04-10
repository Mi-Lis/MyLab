from abc import ABCMeta, abstractmethod


class IbaseLogic:
    __metaclass__ = ABCMeta
    @abstractmethod
    def active(self):
        pass
    @abstractmethod
    def deactive(self):
        pass
    pass