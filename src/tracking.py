"""Vehicle class should be responsible for holding vehicle state and knowing how
to update it (i.e. state updating functions). VehicleTracker class should be
responsible for instantiating and deleting vehicle instances, as well as knowing
when a vehicle should update its state and with what values

basically the vehicle instance is responsible for understanding it's internal
representation based on the external values it's being fed

NOTE:
-   consider playing around with the period attribute of the tracker in case
    optimizations are needed (like if we use large models for detections/classifications)

TODO:
-   maybe add option to read in ROI and direction coordinates from a file for vehicle
    tracker setup
"""

import datetime
from typing import List, Dict, Tuple, Any

import pandas as pd
import numpy as np
import torch

import cv2 as cv
import norfair
from norfair import Tracker

from src.utils.geometry import points_to_rect
from src.utils.image import (
    get_color_from_RGB,
    compute_avg_color,
)
from src.utils.video import VideoHandler

VEHICLE_DATA_COLUMNS = {
    "timestamp": "string",
    "video_timestamp": "string",
    "initial_frame_index": "UInt64",
    "num_of_frames": "UInt64",
    "class": "string",
    "confidence": "Float64",
    "entry_direction": "string",
    "exit_direction": "string",
    "color": "string",
}


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


class _VehicleInstance:
    """class responsible for holding a vehicles state throughout its lifespan

    NOTE: this should only every be instantiated by a vehicle tracker object

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
        class_map: Dict[int, str],
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
        self._color = (CumulativeAverage(), CumulativeAverage(), CumulativeAverage())
        self._entry_direction = None
        self._current_direction = None

        self._class_map = class_map

        self._class_estimate_bins = {
            class_num: 0 for class_num in self._class_map.keys()
        }
        self._class_confidence_bins = {
            class_num: CumulativeAverage() for class_num in self._class_map.keys()
        }

    def get_class(self) -> int:
        """returns the class num of the best estimate for the vehicle class

        Returns
        -------
        int:
            the best estimate class num
        """
        # NOTE: may have to add tie handling
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
            "class": self._class_map[cls],
            "confidence": round(conf, 2),
            "entry_direction": self._entry_direction,
            "exit_direction": self._current_direction,
            "color": get_color_from_RGB(rgb_tuple),
        }

        return data

    def update_direction(self, direction: str):
        """updates the current direction, and if entry direction has not
        been set it will set that in here
        """
        if self._entry_direction is None:
            self._entry_direction = direction
        self._current_direction = direction

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

    def update_color(self, rgb: Tuple[int, int, int]):
        """updates the color estimate for this vehicle"""
        self._color[0].update(rgb[0])
        self._color[1].update(rgb[1])
        self._color[2].update(rgb[2])

    def increment_detection_count(self):
        """increments the number of detections for this vehicle by 1"""
        self._num_of_detections += 1


class VehicleInstanceTracker:
    """Acts as a wrapper around a norfair.tracking.Tracker object, allowing
    for vehicle state estimation and tracking.

    NOTE: currently I'm just holding on to the data and allowing it
    to be extracted into a dataframe from outside of the instance, but if
    the memory usage for storing all vehicles is too much I may have to change
    this to garbage collect dead vehicles while running.
    """

    def __init__(
        self,
        video_handler: VideoHandler,
        class_map: Dict[int, str],
        distance_function,
        distance_threshold,
        roi: List[Tuple[int, int]] | None = None,
        zone_mask: np.ndarray = None,
        zone_mask_map: Dict[int, str] = None,
        initial_datetime: datetime.datetime | None = None,
        hit_counter_max: int = 5,
        initialization_delay: int = 5,
    ):
        self._video_handler = video_handler
        self._class_map = class_map
        self._roi = roi
        self._initial_datetime = initial_datetime
        self._zone_mask = zone_mask
        self._zone_mask_map = zone_mask_map

        self._tracker = Tracker(
            distance_function=distance_function,
            distance_threshold=distance_threshold,
            hit_counter_max=hit_counter_max,
            initialization_delay=initialization_delay,
        )
        self._vehicles: Dict[int, _VehicleInstance] = {}
        self._results: Dict[int, Dict[str, Any]] = {}

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

                # initialize new vehicle instance
                self._vehicles[obj.global_id] = _VehicleInstance(
                    initial_dt=self._initial_datetime,
                    elapsed_time=elapsed_time,
                    initial_frame_index=self._video_handler.current_frame,
                    class_map=self._class_map,
                )

                # # FOR TESTING ............................................
                # cv.circle(
                #     img,
                #     points_to_rect(obj.last_detection.points)
                #     .center.to_int()
                #     .as_tuple(),
                #     5,
                #     (0, 0, 255),
                #     -1,
                # )
                # draw_zones(img, self._zone_mask)
                # cv.imshow("entry", img)
                # cv.waitKey(0)

            # updating tracked vehicle data
            self._vehicles[obj.global_id].increment_detection_count()
            self._update_vehicle_direction(obj)
            self._update_vehicle_class_(obj)
            self._update_vehicle_color(img, obj)

        # purge any vehicles no longer being tracked and store data in results
        for global_id, vehicle in list(self._vehicles.items()):
            if global_id not in [obj.global_id for obj in tracked_objects]:
                self._results[global_id] = vehicle.get_data()
                del self._vehicles[global_id]

        return tracked_objects

    def get_results(self) -> pd.DataFrame:
        """aggregates all tracked vehicle data into a dataframe

        NOTE: may want to do this when a vehicle dies and not all at once at the
        end (if we end up having a LOT of vehicles appear this could get memory
        intensive storing it all the whole time)

        Returns
        -------
        pd.DataFrame:
            the dataframe containing a row of data for each vehicle instance
        """
        return pd.DataFrame(
            [data for _, data in self._results.items()],
            columns=VEHICLE_DATA_COLUMNS,
        ).astype(dtype=VEHICLE_DATA_COLUMNS)

    def _update_vehicle_color(
        self, img: np.ndarray, tracked_obj: norfair.tracker.TrackedObject
    ):
        """computes average color of last detection for this vehicle and then
        updates its state
        """
        bbox = points_to_rect(tracked_obj.last_detection.points)
        avg_color = compute_avg_color(img, bbox, 0.20)
        self._vehicles[tracked_obj.global_id].update_color(avg_color)

    def _update_vehicle_class_(self, tracked_obj: norfair.tracker.TrackedObject):
        """updates the class estimate using the last detection matched with the
        tracked vehicle
        """
        assert isinstance(
            tracked_obj.last_detection.data, dict
        ), "need data dict for detections"

        class_num = tracked_obj.last_detection.data.get("class", None)
        conf = tracked_obj.last_detection.data.get("conf", None)

        bbox = points_to_rect(tracked_obj.last_detection.points)
        weight = bbox.area
        normalizer = (
            self._video_handler.resolution[0] * self._video_handler.resolution[1]
        )

        self._vehicles[tracked_obj.global_id].update_class_estimate(
            class_num, conf, weight, normalizer
        )

    def _update_vehicle_direction(
        self, tracked_obj: norfair.tracker.TrackedObject
    ) -> str:
        """estimates the direction the vehicle is currently at based on the zone
        mask and zone mask map
        """
        center_pt = points_to_rect(tracked_obj.last_detection.points).center.to_int()
        self._vehicles[tracked_obj.global_id].update_direction(
            self._zone_mask_map[self._zone_mask[center_pt.y, center_pt.x]]
        )
