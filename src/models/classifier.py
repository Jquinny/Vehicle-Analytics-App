"""module for classifier implementations"""

import random
from typing import Tuple, Dict

import numpy as np

from src.models.base_model import BaseModel


class TestingClassifier(BaseModel):
    """classifier solely used for testing data aggregation"""

    def setup(self, model_path: str, params: dict):
        pass

    def inference(self, img: np.ndarray, **runtime_args) -> Tuple[int, float]:
        pass

    def get_classes(self) -> Dict[int, str | None]:
        pass


class YOLOv8Classifier(BaseModel):
    """yolov8 classifier"""

    def setup(self, model_path: str, params: dict):
        pass

    def inference(self, img: np.ndarray, **runtime_args) -> Tuple[int, float]:
        pass

    def get_classes(self) -> Dict[int, str | None]:
        pass


class ResNetClassifier(BaseModel):
    """ResNet classifier"""

    def setup(self, model_path: str, params: dict):
        pass

    def inference(self, img: np.ndarray, **runtime_args) -> Tuple[int, float]:
        pass

    def get_classes(self) -> Dict[int, str | None]:
        pass


class ViTClassifier(BaseModel):
    """Visual Transformer classifier"""

    def setup(self, model_path: str, params: dict):
        pass

    def inference(self, img: np.ndarray, **runtime_args) -> Tuple[int, float]:
        pass

    def get_classes(self) -> Dict[int, str | None]:
        pass


class ClipClassifier(BaseModel):
    """OpenAI Clip classifier"""

    def setup(self, model_path: str, params: dict):
        pass

    def inference(self, img: np.ndarray, **runtime_args) -> Tuple[int, float]:
        pass

    def get_classes(self) -> Dict[int, str | None]:
        pass
