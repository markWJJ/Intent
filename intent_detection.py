from abc import ABCMeta,abstractmethod



class Intent_Detection(metaclass=ABCMeta):
    def __init__(self,**kwargs):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def infer(self):
        pass


class b(Intent_Detection):

    def __init__(self,**kwargs):
        pass


if __name__ == '__main__':
    bb=b()




