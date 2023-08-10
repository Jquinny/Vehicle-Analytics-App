"""module for object detector implementations"""


from typing import List

import norfair
import numpy as np
import torch

from norfair import Detection
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
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.inf_params = params.get("inference_params", {})

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
        pass

    def inference(
        self, img: np.ndarray, **runtime_args
    ) -> List[norfair.tracker.Detection]:
        pass


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
