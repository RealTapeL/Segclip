from abc import ABC, abstractmethod

class BaseClassifier(ABC):
    @abstractmethod
    def classify(self, image, masks):
        pass

    @abstractmethod
    def preprocess(self, image):
        pass

    @abstractmethod
    def load_model(self, config):
        pass