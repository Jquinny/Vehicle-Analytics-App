"""Vehicle class should be responsible for holding vehicle state and knowing how
to update it (i.e. state updating functions). VehicleTracker class should be
responsible for instantiating and deleting vehicle instances, as well as knowing
when a vehicle should update its state and with what values

basically the vehicle instance is responsible for understanding it's internal
representation based on the external values it's being fed

NOTE:
-   possibly do vehicle garbage collection when we get to 1000 vehicle instances
-   consider playing around with the period attribute of the tracker in case
    optimizations are needed (like if we use large models for detections/classifications)

TODO:
-   add option to read in ROI and direction coordinates from a file for vehicle
    tracker setup
-   figure out how to garbage collect the vehicle instances efficiently
-   to compute the entry and exit directions, use the velocity estimate of the
    tracked objects to see which direction coordinate (VELOCITY WILL NOT WORK)
        -   if that doesn't work, could try fitting a line through it's
            trajectory (would require storing points from each detection)
            and then estimating based on that
"""

import random
from copy import deepcopy
import datetime
from typing import List, Dict, Tuple, Any

import pandas as pd
import numpy as np
import torch

import cv2 as cv
import norfair
from norfair import Tracker, Detection, Paths, Video

from config import (
    CLASSIFIER_NUM2NAME,
    CLASSIFIER_NAME2NUM,
    VEHICLE_DATA_COLUMNS,
    DETECTOR_NUM2NAME,
)
from src.utils.drawing import draw_vector
from src.utils.geometry import Rect
from src.utils.image import (
    get_color_from_RGB,
    compute_avg_color,
)
from src.utils.video import VideoHandler


class CumulativeAverage:
    """Basic class for computing cumulative averages so we don't have to store
    lists of results for each detection and do sums at the end
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

    algorithm for bins is as follows (use this when just using single shot
    i.e. no classifier):
    For each frame the object is detected in, do:
    1. compute bounding box area (or distance to center of image), call this w
    2. add (w * class_conf) / full_img_area to class_estimate_bins[class_num]
    TODO: update to better math when you have time to think about it
    -   ALSO: add classes as a parameter to __init__(). we can grab this from the
        model and then build our bins from it instead of the hardcoded config
    """

    def __init__(
        self,
        initial_dt: datetime.datetime,
        elapsed_time: int,
        initial_frame_index: int,
        entry_dir: str,
    ):
        """Initialises the state for a vehicle"""

        self._num_of_detections = 0
        self._timestamp = (
            str(initial_dt + datetime.timedelta(seconds=elapsed_time))
            if initial_dt
            else None
        )
        self._video_timestamp = str(datetime.timedelta(seconds=elapsed_time))
        self._initial_frame_index = initial_frame_index
        self._entry_direction = entry_dir

        self._speed = CumulativeAverage()
        self._color = (CumulativeAverage(), CumulativeAverage(), CumulativeAverage())
        self._exit_direction = None

        self._class_estimate_bins = {
            class_num: 0 for class_num in CLASSIFIER_NUM2NAME.keys()
        }
        self._class_confidence_bins = {
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
        return max(self._class_estimate_bins, key=self._class_estimate_bins.get)

    def get_class_confidence(self) -> float:
        """returns the confidence for the best estimate of the vehicle class

        Returns
        -------
        float:
            the confidence of the best estimate class
        """
        cls_num = self.get_class()
        return float(self._class_confidence_bins[cls_num])

    def get_data(self) -> Dict[str, Any]:
        """returns this vehicle instance's data dictionary

        Returns
        -------
        Dict[str, Any]:
            the vehicle data dictionary
        """
        cls = self.get_class()
        conf = self.get_class_confidence()
        rgb_tuple = tuple(map(int, self._color))

        data = {
            "timestamp": self._timestamp,
            "video_timestamp": self._video_timestamp,
            "initial_frame_index": self._initial_frame_index,
            "num_of_frames": self._num_of_detections,
            "class": DETECTOR_NUM2NAME[cls],  # NOTE: change this later
            "confidence": round(conf, 2),
            "entry_direction": self._entry_direction,
            "exit_direction": self._exit_direction,
            "speed": float(self._speed),
            "color": get_color_from_RGB(rgb_tuple),
        }

        return data

    def update_class_estimate(
        self,
        class_num: int,
        class_conf: float,
        weight: float | int,
        normalizer: float | int,
    ):
        """updates the class estimate and confidence bins based on model inference"""
        self._class_estimate_bins[class_num] += (weight * class_conf) / normalizer
        self._class_confidence_bins[class_num].update(class_conf)

    def update_speed(self, speed: float):
        """updates the speed estimate for this vehicle"""
        self._speed.update(speed)

    def update_color(self, rgb: Tuple[int, int, int]):
        """updates the color estimate for this vehicle"""
        self._color[0].update(rgb[0])
        self._color[1].update(rgb[1])
        self._color[2].update(rgb[2])

    def increment_detection_count(self):
        """increments the number of detections for this vehicle by 1

        NOTE: possibly change this to take in an `increment` amount that we
        increment by instead of always 1
        """
        self._num_of_detections += 1


class VehicleInstanceTracker:
    """class responsible for producing and destroying vehicle instances properly.
    Acts as a wrapper around a norfair.tracking.Tracker object

    When a vehicle instance is destroyed, the vehicle data should be added
    to the results dataframe

    TODO: use LRU caching system for garbage collecting vehicle instances maybe?
    """

    def __init__(
        self,
        video_handler: VideoHandler,
        distance_function,
        distance_threshold,
        roi: List[Tuple[int, int]] | None = None,
        direction_coords: Dict[str, Tuple[int, int]] | None = None,
        initial_datetime: datetime.datetime | None = None,
    ):
        self._video_handler = video_handler
        self._roi = roi
        self._direction_coords = direction_coords
        self._initial_datetime = initial_datetime

        self._tracker = Tracker(distance_function, distance_threshold)
        self._vehicles: Dict[int, VehicleInstance] = {}

        self._results = pd.DataFrame(columns=VEHICLE_DATA_COLUMNS).astype(
            VEHICLE_DATA_COLUMNS
        )

    def _check_vehicle_status(self, vehicle: VehicleInstance):
        """check what state a vehicle object is currently in"""
        pass

    def update(
        self,
        img: np.ndarray,
        detections: List[norfair.tracker.Detection],
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
        tracked_objects = self._tracker.update(detections)

        for obj in tracked_objects:
            if obj.global_id not in self._vehicles.keys():
                elapsed_time = int(
                    self._video_handler.current_frame / self._video_handler.fps
                )

                # TODO: compute entry direction from coordinates and initial
                x1, y1 = obj.get_estimate().astype(int)[0]
                x2, y2 = obj.get_estimate().astype(int)[1]
                center_x, center_y = (
                    Rect(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
                    .center.to_int()
                    .as_tuple()
                )
                print(f"Entry Coordinate: ({center_x}, {center_y})")

                # initialize new vehicle instance
                self._vehicles[obj.global_id] = VehicleInstance(
                    initial_dt=self._initial_datetime,
                    elapsed_time=elapsed_time,
                    initial_frame_index=self._video_handler.current_frame,
                    entry_dir=None,  # TODO
                )

            self._vehicles[obj.global_id].increment_detection_count()
            self._update_vehicle_color(img, obj)
            # self._update_vehicle_speed(obj)  # TODO: implement this logic
            self._update_vehicle_class_estimate(obj)

            print(self._vehicles[obj.global_id].get_data())

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
        for global_id, vehicle in self._vehicles.items():
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
        bbox = Rect(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
        avg_color = compute_avg_color(img, bbox, 0.20)
        self._vehicles[tracked_obj.global_id].update_color(avg_color)

    def _update_vehicle_speed(self, tracked_obj: norfair.tracker.TrackedObject):
        # TODO: compute speed estimate based on norfair velocity estimate somehow
        self._vehicles[tracked_obj.global_id].update_speed(2)

    def _update_vehicle_class_estimate(
        self, tracked_obj: norfair.tracker.TrackedObject
    ):
        """updates the class estimate using the last detection matched with the
        tracked vehicle
        """
        assert isinstance(
            tracked_obj.last_detection.data, dict
        ), "need data dict for detections"

        class_num = tracked_obj.last_detection.data.get("class", None)
        conf = tracked_obj.last_detection.data.get("conf", None)

        x1, y1 = tracked_obj.last_detection.points[0]
        x2, y2 = tracked_obj.last_detection.points[1]
        bbox = Rect(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
        weight = bbox.area
        normalizer = (
            self._video_handler.resolution[0] * self._video_handler.resolution[1]
        )

        self._vehicles[tracked_obj.global_id].update_class_estimate(
            class_num, conf, weight, normalizer
        )
