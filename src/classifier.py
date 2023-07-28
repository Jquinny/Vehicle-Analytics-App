import random
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from config import CLASSIFIER_NUM2NAME


class Classifier(ABC):
    """abstract base class for defining classifier objects"""

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def inference(self, img: np.ndarray) -> Tuple[int, float]:
        pass


class TestingClassifier(Classifier):
    """classifier solely used for testing data aggregation"""

    def setup_model(self):
        pass

    def inference(self, img: np.ndarray) -> Tuple[int, float]:
        label = random.choice(list(CLASSIFIER_NUM2NAME.keys()))
        conf = random.random()

        return label, conf


class YOLOv8Classifier(Classifier):
    """yolov8 classifier"""

    def __init__(self, params: dict = None):
        self.params = params

    def setup_model(self):
        pass

    def inference(self, img: np.ndarray) -> Tuple[int, float]:
        pass


class ResNetClassifier(Classifier):
    """ResNet classifier"""

    def __init__(self, params: dict = None):
        self.params = params

    def setup_model(self):
        pass

    def inference(self, img: np.ndarray) -> Tuple[int, float]:
        pass


class ViTClassifier(Classifier):
    """Visual Transformer classifier"""

    def __init__(self, params: dict = None):
        pass

    def setup_model(self):
        pass

    def inference(self, img: np.ndarray) -> Tuple[int, float]:
        pass


class ClipClassifier(Classifier):
    """OpenAI Clip classifier"""

    def __init__(self, params: dict = None):
        pass

    def setup_model(self):
        pass

    def inference(self, img: np.ndarray) -> Tuple[int, float]:
        pass
