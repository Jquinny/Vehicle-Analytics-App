"""Vehicle class should be responsible for holding vehicle state and knowing how
to update it (i.e. state updating functions). VehicleTracker class should be
responsible for instantiating and deleting vehicle instances, as well as knowing
when a vehicle should update its state"""

import random
from copy import deepcopy
from typing import List, Dict, Tuple, Any

import pandas as pd
import numpy as np
import torch

import cv2 as cv
import norfair
from norfair import Tracker, Detection, Paths, Video
from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)

from config import CLASSIFIER_NUM2NAME, CLASSIFIER_NAME2NUM, VEHICLE_DATA_COLUMNS
from src.utils.image import bbox_center


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
        return float(self.average)

    def __int__(self):
        return int(self.average)

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

    def __init__(self):
        """Initialises the state for a vehicle"""
        self.num_of_detections = 0
        self.time_of_day = None
        self.time_in_video = None
        self.speed = CumulativeAverage()
        self.color = (CumulativeAverage(), CumulativeAverage(), CumulativeAverage())
        self.entry_direction = None
        self.exit_direction = None

        self.class_estimate_bins = {
            class_num: 0 for class_num in CLASSIFIER_NUM2NAME.keys()
        }
        self.class_confidence_bins = {
            class_num: CumulativeAverage() for class_num in CLASSIFIER_NUM2NAME.keys()
        }

    def get_class(self) -> int:
        """returns the class num of the best estimate for the vehicle class

        Returns
        -------
        int:
            the best estimate class num
        """
        # NOTE: may have to handle ties somehow
        return max(self.class_estimate_bins, key=self.class_estimate_bins.get)

    def get_class_confidence(self) -> float:
        """returns the confidence for the best estimate of the vehicle class

        Returns
        -------
        float:
            the confidence of the best estimate class
        """
        cls_num = self.get_class()
        return float(self.class_confidence_bins[cls_num])

    def _get_color_from_RGB(self) -> str:
        """returns the human readable color name from the RGB color tuple associated
        with this vehicle

        Returns
        -------
        str:
            the human readable color name
        """
        rgb_tuple = tuple(map(int, self.color))

        # a dictionary of all the hex and their respective names in css3
        css3_db = CSS3_HEX_TO_NAMES
        names = []
        rgb_values = []
        for color_hex, color_name in css3_db.items():
            names.append(color_name)
            rgb_values.append(hex_to_rgb(color_hex))

        kdt_db = KDTree(rgb_values)
        _, index = kdt_db.query(rgb_tuple)
        return names[index]

    def get_data(self) -> Dict[str, Any]:
        """returns this vehicle instance's data dictionary

        Returns
        -------
        Dict[str, Any]:
            the vehicle data dictionary
        """
        cls = self.get_class()
        conf = self.get_class_confidence()

        data = {
            "num_of_frames": self.num_of_detections,
            "time_of_day": self.time_of_day,
            "time_in_video": self.time_in_video,
            "class": cls,
            "confidence": conf,
            "entry_direction": self.entry_direction,
            "exit_direction": self.exit_direction,
            "speed": float(self.speed),
            "color": self._get_color_from_RGB(),
        }

        return data

    def set_time_of_day(self, initial_datetime: str, elapsed_time: float):
        """sets the time of day based on an initial datetime string and the elapsed
        video time

        Arguments
        ---------
        initial_datetime (str):
            the initial datetime at the start of the video
        elapsed_time (float):
            the elapsed video time
        """
        pass

    def set_time_in_video(self, elapsed_time: float):
        """sets the initial detection time with respect to elapsed video time

        Arguments
        ---------
        elapsed_time (float):
            the elapsed video time
        """
        pass

    def update_class_estimate(self, new_value):
        """updates the class estimate bins based on model inference"""
        pass

    def update_class_conf(self, new_value):
        """updates the class confidence bins based on model inference"""
        pass

    def update_speed(self, speed_estimate: float):
        """updates the speed estimate for this vehicle

        Arguments
        ---------
        speed_estimate (float):
            the estimated speed of the vehicle during the last detection
        """
        self.speed.update(speed_estimate)

    def update_color(self, rgb: Tuple[int, int, int]):
        """updates the color estimate for this vehicle

        Arguments
        ---------
        rgb (Tuple[int, int, int]):
            the average rgb tuple computed for the last vehicle detection
        """
        self.color[0].update(rgb[0])
        self.color[1].update(rgb[1])
        self.color[2].update(rgb[2])

    def increment_frame_count(self):
        """increments the number of detections for this vehicle by 1

        NOTE: possibly change this to take in an `increment` amount that we
        increment by instead of always 1
        """
        self.data["num_of_frames"] += 1


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

    def _check_vehicle_status(self, vehicle: VehicleInstance):
        """check what state a vehicle object is currently in"""
        pass

    def update(
        self, img: np.ndarray, detections: List[norfair.tracker.Detection]
    ) -> List[norfair.tracker.TrackedObject]:
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
                # update vehicle color
                # x1, y1 = obj.last_detection.points[0]
                # x2, y2 = obj.last_detection.points[1]
                # avg_color = self.compute_avg_color(img, x1, y1, x2, y2)
                # self.vehicles[obj.global_id].update_color(avg_color)
                self._update_vehicle_color(img, obj)

                # update vehicle speed TODO
                self.vehicles[obj.global_id].update_speed(2)
                # self.vehicles[obj.global_id].update_speed()
                # self.vehicles[obj.global_id].update_speed()
                # self.vehicles[obj.global_id].update_speed()

                print(self.vehicles[obj.global_id].get_data())
            else:
                # initialize new vehicle instance
                assert obj.global_id not in self.vehicles.keys()
                self.vehicles[obj.global_id] = VehicleInstance()

                # TODO: do state initialization like entry point and datetime
                # + video timestamps

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

    def _update_vehicle_color(
        self, img: np.ndarray, tracked_obj: norfair.tracker.TrackedObject
    ):
        """computes average color of last detection for this vehicle and then
        updates its state
        """
        x1, y1 = tracked_obj.last_detection.points[0]
        x2, y2 = tracked_obj.last_detection.points[1]
        avg_color = self._compute_avg_color(img, x1, y1, x2, y2)
        self.vehicles[tracked_obj.global_id].update_color(avg_color)

    def _compute_avg_color(
        self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> Tuple[int, int, int]:
        """computes the average color within a bounding box by computing the
        average color in a square near the center of the bounding box
        (since the road overtakes the color value in most bounding boxes)
        """
        area_pct = 0.20

        center_x, center_y = bbox_center(x1, y1, x2, y2)

        new_x1 = int(center_x - area_pct * (x2 - x1) // 2)
        new_y1 = int(center_y - area_pct * (y2 - y1) // 2)
        new_x2 = int(center_x + area_pct * (x2 - x1) // 2)
        new_y2 = int(center_y + area_pct * (y2 - y1) // 2)

        # using opencv for images so bgr instead of rgb
        b = int(np.mean(img[new_y1:new_y2, new_x1:new_x2, 0]))
        g = int(np.mean(img[new_y1:new_y2, new_x1:new_x2, 1]))
        r = int(np.mean(img[new_y1:new_y2, new_x1:new_x2, 2]))

        return (r, g, b)


# NOTE: note quite sure what this was for
def center(points):
    return [np.mean(np.array(points), axis=0)]


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "bbox"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""

    norfair_detections: List[Detection] = []

    for detection in yolo_detections[0].boxes:
        bbox_as_xyxy = detection.cpu().numpy().xyxy.astype(int)
        # score = detection.boxes.cpu().numpy().conf
        # cls = detection.boxes.cpu().numpy().cls.astype(int)
        if track_points == "centroid":
            x1 = bbox_as_xyxy[0, 0]
            y1 = bbox_as_xyxy[0, 1]
            x2 = bbox_as_xyxy[0, 2]
            y2 = bbox_as_xyxy[0, 3]

            center_x, center_y = bbox_center(x1, y1, x2, y2)

            centroid = np.array([center_x, center_y])
            norfair_detections.append(Detection(points=centroid))
        elif track_points == "bbox":
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
