# TODO: add type hints

from typing import List, Optional, Union, NewType

import pandas as pd
import numpy as np
import torch

import cv2 as cv
import norfair
from norfair import Detection, Paths, Tracker, Video

from config import CLASSIFIER_NUM2NAME, DETECTOR_NUM2NAME, VEHICLE_DATA_COLUMNS


class VehicleInstance:
    """class responsible for holding a vehicles state throughout its lifespan

    TODO: figure out clean implementation for updating the class_estimate_bins
    and any other state that I need to track

    algorithm for bins is as follows:
    For each frame the object is detected in, do:
    1. compute bounding box area (or distance to center of image), call this w
    2. add (w * class_conf) / full_img_area to class_estimate_bins[class_num]
    """

    def __init__(self, tracked_object: norfair.tracker.TrackedObject):
        # TODO: figure out what extra state needs to be tracked
        self.tracked_object = tracked_object
        self.class_estimate_bins = {
            class_num: 0 for class_num in CLASSIFIER_NUM2NAME.keys()
        }


class VehicleInstanceTracker:
    """class responsible for producing and destroying vehicle instances properly.

    When a vehicle instance is destroyed, the vehicle data should be added
    to the results dataframe
    """

    def __init__(self):
        self.results = pd.DataFrame(columns=VEHICLE_DATA_COLUMNS).astype(
            VEHICLE_DATA_COLUMNS
        )
        self.vehicles = List[VehicleInstance]  # maybe use a set for this ?

    def check_vehicle_status(self, vehicle: VehicleInstance):
        """check what state a vehicle object is currently in"""
        pass

    def update_vehicle_states(
        self, tracked_objects: List[norfair.tracker.TrackedObject]
    ):
        """updates the states of all vehicle instances"""
        pass

    def get_vehicle_data(self, vehicle: VehicleInstance):
        """aggregates tracked vehicle data once it is destroyed"""


# NOTE: note quite sure what this was for
def center(points):
    return [np.mean(np.array(points), axis=0)]


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    # TODO: figure this out
    # if track_points == "centroid":
    # for detection in yolo_detections:
    #     bbox_as_xywh = detection.boxes.cpu().numpy().xywh.astype(int)
    #     score = detection.boxes.cpu().numpy().conf
    #     cls = detection.boxes.cpu().numpy().cls.astype(int)
    #     if len(bbox_as_xywh) > 0:
    #         centroid = np.array(
    #             [
    #                 bbox_as_xywh[0, 0] + bbox_as_xywh[0, 2] // 2,
    #                 bbox_as_xywh[0, 1] + bbox_as_xywh[0, 3] // 2,
    #             ]
    #         )
    #         norfair_detections.append(
    #             Detection(
    #                 points=centroid,
    #                 scores=score,
    #                 label=LABEL2NAME[cls[0]],
    #             )
    #         )
    # elif track_points == "bbox":
    for detection in yolo_detections[0].boxes:
        bbox_as_xyxy = detection.cpu().numpy().xyxy.astype(int)
        score = detection.cpu().numpy().conf
        cls = detection.cpu().numpy().cls.astype(int)
        bbox = np.array(
            [
                [bbox_as_xyxy[0, 0], bbox_as_xyxy[0, 1]],
                [bbox_as_xyxy[0, 2], bbox_as_xyxy[0, 3]],
            ]
        )
        score_per_corner = np.ones(2, dtype=int) * score
        norfair_detections.append(
            Detection(
                points=bbox,
                scores=score_per_corner,
                label=DETECTOR_NUM2NAME[cls[0]],
            )
        )

    return norfair_detections


def merge_frames(track_mask, video_frame):
    return cv.addWeighted(track_mask, 1, video_frame, 1, 0)
