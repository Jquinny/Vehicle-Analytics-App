"""NOTE: possibly do all necessary detection format conversions inside each
class so I don't have to do them separately in the tracking.py file"""

from abc import ABC, abstractmethod
from typing import List

import norfair
import numpy as np


class Detector(ABC):
    """abstract base class for defining detector objects"""

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def inference(self, img) -> List[norfair.tracker.Detection]:
        pass


class YOLOv8Detector(Detector):
    """yolov8 object detector"""

    def __init__(self, params: dict = None):
        self.params = params

    def setup_model(self):
        from ultralytics import YOLO

        # TODO: set up model somehow

    def inference(self, img: np.ndarray) -> List[norfair.tracker.Detection]:
        pass


class RTDETRDetector(Detector):
    """RT-DETR object detector"""

    def __init__(self, params: dict = None):
        pass

    def setup_model(self):
        pass

    def inference(self, img: np.ndarray) -> List[norfair.tracker.Detection]:
        pass


class FasterRCNNDetector(Detector):
    """Faster-RCNN object detector"""

    def __init__(self, params: dict = None):
        pass

    def setup_model(self):
        pass

    def inference(self, img: np.ndarray) -> List[norfair.tracker.Detection]:
        pass


class EfficientDetDetector(Detector):
    """EfficientDet object detector"""

    def __init__(self, params: dict = None):
        pass

    def setup_model(self):
        pass

    def inference(self, img: np.ndarray):
        pass
