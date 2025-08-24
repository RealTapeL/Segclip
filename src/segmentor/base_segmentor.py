from abc import ABC, abstractmethod
import numpy as np

class BaseSegmentor(ABC):
    @abstractmethod
    def segment(self, image):
        pass

    @abstractmethod
    def preprocess(self, image):
        pass

    @abstractmethod
    def postprocess(self, result):
        pass