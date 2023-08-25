"""module for classifier implementations"""

import random
from typing import Tuple, Dict, Any

import numpy as np

from ultralytics import YOLO

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
        """loads the model from the file path and initializes it with any
        necessary parameters provided in the params dict

        Arguments
        ---------
        model_path (str):
            absolute path to the model weights
        params (dict):
            dictionary of possible hyperparameters required for model setup"""
        self.model = YOLO(model_path)
        self.classes: Dict[int, str] | {} = (
            self.model.names if self.model.names else params.get("classes", {})
        )
        self.metrics: Dict[str, Any] | {} = params.get("metrics", {})
        self.inf_params: Dict[str, Any] | {} = params.get("inference_params", {})

    def inference(self, img: np.ndarray, **runtime_args) -> Tuple[int, float]:
        """run inference on an image using the models stored inference
        parameters (if any) as well as inference parameters given at runtime
        (if any)

        Arguments
        ---------
        img (np.ndarray):
            the image to run inference on
        **runtime_args:
            any keyword arguments to be used as inference parameters
        """
        self.inf_params.update(runtime_args)
        if self.inf_params:
            results = self.model.predict(img, **self.inf_params)
        else:
            # no inference params specified, using defaults
            results = self.model.predict(img)

        print(results)

        return results

    def get_classes(self) -> Dict[int, str | None]:
        return self.classes


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
