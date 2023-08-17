"""module for object detector implementations

NOTE: each detector must return a list of norfair detection objects. Those
detection objects should be instantiated with a data dictionary in the form
data = {"class": <int>, "conf": <float>}
"""


from typing import List, Dict, Any

import norfair
import numpy as np
import torch

from norfair import Detection
from ultralytics import RTDETR, YOLO

from src.models.base_model import BaseModel


class YOLOv8Detector(BaseModel):
    """yolov8 object detector"""

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
        self.classes: Dict[int, str] | None = (
            self.model.names if self.model.names else params.get("classes", None)
        )
        self.metrics: Dict[str, Any] | None = params.get("metrics", None)
        self.inf_params: Dict[str, Any] | None = params.get("inference_params", None)

    def inference(
        self, img: np.ndarray, **runtime_args
    ) -> List[norfair.tracker.Detection]:
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

        return self._to_norfair(results)

    def get_classes(self) -> Dict[int, str | None]:
        return self.classes

    def _to_norfair(
        self,
        yolo_detections: torch.tensor,
    ) -> List[Detection]:
        """convert detections_as_xywh to norfair detections"""

        norfair_detections: List[Detection] = []

        for detection in yolo_detections[0].boxes:
            bbox_as_xyxy = detection.cpu().numpy().xyxy.astype(int)
            score = detection.cpu().numpy().conf.item()
            cls = detection.cpu().numpy().cls.item()
            bbox = np.array(
                [
                    [bbox_as_xyxy[0, 0], bbox_as_xyxy[0, 1]],
                    [bbox_as_xyxy[0, 2], bbox_as_xyxy[0, 3]],
                ]
            )
            norfair_detections.append(
                Detection(points=bbox, data={"class": cls, "conf": score})
            )

        return norfair_detections


class RTDETRDetector(BaseModel):
    """RT-DETR object detector"""

    def setup(self, model_path: str, params: dict):
        """loads the model from the file path and initializes it with any
        necessary parameters provided in the params dict

        Arguments
        ---------
        model_path (str):
            absolute path to the model weights
        params (dict):
            dictionary of possible hyperparameters required for model setup"""
        self.model = RTDETR(model_path)
        self.classes: Dict[int, str] | None = (
            self.model.names if self.model.names else params.get("classes", None)
        )
        self.metrics: Dict[str, Any] | None = params.get("metrics", None)
        self.inf_params: Dict[str, Any] | None = params.get("inference_params", None)

    def inference(
        self, img: np.ndarray, **runtime_args
    ) -> List[norfair.tracker.Detection]:
        self.inf_params.update(runtime_args)
        if self.inf_params:
            results = self.model.predict(img, **self.inf_params)
        else:
            # no inference params specified, using defaults
            results = self.model.predict(img)

        return self._to_norfair(results)

    def get_classes(self) -> Dict[int, str | None]:
        return self.classes

    def _to_norfair(
        self,
        rtdetr_detections: torch.tensor,
    ) -> List[Detection]:
        """convert detections_as_xywh to norfair detections"""

        norfair_detections: List[Detection] = []

        for detection in rtdetr_detections[0].boxes:
            bbox_as_xyxy = detection.cpu().numpy().xyxy.astype(int)
            score = detection.cpu().numpy().conf.item()
            cls = detection.cpu().numpy().cls.item()
            bbox = np.array(
                [
                    [bbox_as_xyxy[0, 0], bbox_as_xyxy[0, 1]],
                    [bbox_as_xyxy[0, 2], bbox_as_xyxy[0, 3]],
                ]
            )
            norfair_detections.append(
                Detection(points=bbox, data={"class": cls, "conf": score})
            )

        return norfair_detections


class FasterRCNNDetector(BaseModel):
    """Faster-RCNN object detector"""

    def setup(self, model_path: str, params: dict):
        pass

    def inference(
        self, img: np.ndarray, **runtime_args
    ) -> List[norfair.tracker.Detection]:
        pass


class EfficientDetDetector(BaseModel):
    """EfficientDet object detector"""

    def setup(self, model_path: str, params: dict):
        pass

    def inference(
        self, img: np.ndarray, **runtime_args
    ) -> List[norfair.tracker.Detection]:
        pass
