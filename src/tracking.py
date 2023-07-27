# TODO: add type hints

import random
from copy import deepcopy
from typing import List, Dict, Optional, Union, NewType

import pandas as pd
import numpy as np
import torch

import cv2 as cv
import norfair
from norfair import Tracker, Detection, Paths, Video

from config import CLASSIFIER_NUM2NAME, CLASSIFIER_NAME2NUM, VEHICLE_DATA_COLUMNS


class CumulativeAverage:
    """Basic class for computing cumulative averages so we don't have to store
    lists of results for each detection and do sums at the end for computing
    averages.
    """

    def __init__(self):
        self.average = 0
        self.n = 0

    def update(self, new_value):
        self.n += 1
        self.average += (new_value - self.average) / self.n

    def __float__(self):
        return self.average

    def __repr__(self):
        return "average: " + str(self.average)


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
        """Initialises the state for a vehicle

        Arguments
        ---------
        tracked_object (norfair.tracker.TrackedObject):
            the TrackedObject associated with this vehicle
        """

        self.tracked_object = tracked_object
        self.state = {key: None for key in VEHICLE_DATA_COLUMNS.keys()}
        self.num_of_detections = 0

        self.class_estimate_bins = {
            class_num: 0 for class_num in CLASSIFIER_NUM2NAME.keys()
        }
        self.class_confidence_bins = {
            class_num: CumulativeAverage() for class_num in CLASSIFIER_NUM2NAME.keys()
        }

        self.state["speed"] = 0
        self.state["color"] = (0, 0, 0)

        # TODO: use center of last detection to compute closest coordinate point
        self.entry_pt = None
        self.exit_pt = None

    def update_estimates(self):
        """updates the class estimate bins based on the last detection"""

        detection = self.tracked_object.last_detection
        cls = detection.label
        conf = detection.data["conf"]

        self.class_estimate_bins[""]

    def cumulative_average_update(self, new_value):
        pass

    def color_update(self, new_value):
        pass


class VehicleInstanceTracker:
    """class responsible for producing and destroying vehicle instances properly.
    Acts as a wrapper around a norfair.tracking.Tracker object

    When a vehicle instance is destroyed, the vehicle data should be added
    to the results dataframe
    """

    def __init__(self, distance_function, distance_threshold):
        self.results = pd.DataFrame(columns=VEHICLE_DATA_COLUMNS).astype(
            VEHICLE_DATA_COLUMNS
        )
        self.tracker = Tracker(distance_function, distance_threshold)
        self.vehicles: Dict[int, VehicleInstance] = {}

    def check_vehicle_status(self, vehicle: VehicleInstance):
        """check what state a vehicle object is currently in"""
        pass

    def update(self, detections: List[norfair.tracker.Detection]):
        """updates the states of all active vehicle instances

        Arguments
        ---------
        detections (List[norfair.tracker.Detection]):
            list of norfair detection objects obtained after detection+classification

        Returns
        -------
        List[norfair.tracker.TrackedObject]:
            the list of active objects
        """

        tracked_objects = self.tracker.update(detections)

        for obj in tracked_objects:
            if obj.global_id in self.vehicles.keys():
                # update vehicle state TODO: figure out how to do this well
                # self.vehicles[obj.global_id].update()
                pass
            else:
                # initialize new vehicle instance
                assert obj.global_id not in self.vehicles.keys()
                self.vehicles[obj.global_id] = VehicleInstance(obj)

        return tracked_objects

    def extract_vehicle_data(self) -> pd.DataFrame:
        """aggregates all tracked vehicle data into a dataframe

        NOTE: may want to do this when a vehicle dies and not all at once at the
        end (if we end up having a LOT of vehicles appear this could get memory
        intensive storing it all the whole time)

        Returns
        -------
        pd.DataFrame:
            the dataframe containing a row of data for each vehicle instance
        """

        for global_id, vehicle in self.vehicles.items():
            # TODO: update dataframe
            pass


# NOTE: note quite sure what this was for
def center(points):
    return [np.mean(np.array(points), axis=0)]


# TODO: test centroid tracking to see if it works
def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        for detection in yolo_detections[0].boxes:
            bbox_as_xywh = detection.boxes.cpu().numpy().xywh.astype(int)
            # score = detection.boxes.cpu().numpy().conf
            # cls = detection.boxes.cpu().numpy().cls.astype(int)
            centroid = np.array(
                [
                    bbox_as_xywh[0, 0] + bbox_as_xywh[0, 2] // 2,
                    bbox_as_xywh[0, 1] + bbox_as_xywh[0, 3] // 2,
                ]
            )
            norfair_detections.append(Detection(points=centroid))
    elif track_points == "bbox":
        for detection in yolo_detections[0].boxes:
            bbox_as_xyxy = detection.cpu().numpy().xyxy.astype(int)
            # conf = detection.cpu().numpy().conf
            # cls = detection.cpu().numpy().cls.astype(int)
            bbox = np.array(
                [
                    [bbox_as_xyxy[0, 0], bbox_as_xyxy[0, 1]],
                    [bbox_as_xyxy[0, 2], bbox_as_xyxy[0, 3]],
                ]
            )
            norfair_detections.append(Detection(points=bbox))

    return norfair_detections


def merge_frames(track_mask, video_frame):
    return cv.addWeighted(track_mask, 1, video_frame, 1, 0)
