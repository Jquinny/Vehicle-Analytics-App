from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import datetime

import numpy as np

import cv2 as cv
from norfair import Detection

from src.utils.geometry import points_to_rect, Point, Rect
from src.models.base_model import BaseModel


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


@dataclass
class VehicleDetection:
    """class responsible for holding image and class related info for detections
    of a vehicle instance
    """

    img: np.ndarray
    bbox: Rect
    cls: int
    conf: float
    frame_idx: int


class VehicleInstance:
    """class responsible for holding a vehicles state throughout its lifespan"""

    def __init__(
        self,
        initial_dt: datetime.datetime,
        elapsed_time: int,
        initial_frame_index: int,
        detector_class_map: Dict[int, str],
    ):
        """Initialises the state for a vehicle"""

        self._num_of_frames = 0
        self._timestamp = (
            str(initial_dt + datetime.timedelta(seconds=elapsed_time))
            if initial_dt
            else None
        )
        self._video_timestamp = str(datetime.timedelta(seconds=elapsed_time))
        self._initial_frame_index = initial_frame_index

        self._entry_coord: Point | None = None
        self._current_coord: Point | None = None
        self._entry_direction: str | None = None
        self._exit_direction: str | None = None

        self._detections: List[VehicleDetection] = []

        self._detector_class_map = detector_class_map
        self._detector_class_conf_bins = [
            CumulativeAverage() for _ in range(len(detector_class_map))
        ]

        self._class: str | None = None
        self._class_conf: float | None = None

    def get_class_info(self) -> Tuple[str, float] | Tuple[None, None]:
        """returns the class estimate and confidence for the estimate in the form
        Tuple[str, float] if .classify() has been called, otherwise returns None, None
        """
        return self._class, self._class_conf

    def get_direction_info(self) -> Tuple[str, str] | Tuple[None, None]:
        """returns the entry and exit direction estimates in the form Tuple[entry, exit]
        if .compute_directions() as been called, otherwise returns None, None
        """
        return self._entry_direction, self._exit_direction

    def get_detections(self) -> List[VehicleDetection]:
        """returns the uniformly distributed detections associated with this vehicle"""
        return self._detections

    def get_data(self) -> Dict[str, Any]:
        """returns this vehicle instance's data dictionary

        Returns
        -------
        Dict[str, Any]:
            the vehicle data dictionary
        """

        data = {
            "timestamp": self._timestamp,
            "video_timestamp": self._video_timestamp,
            "initial_frame": self._initial_frame_index,
            "total_frames": self._num_of_frames,
            "class": self._class,
            "confidence": round(self._class_conf, 2),
            "entry_direction": self._entry_direction,
            "exit_direction": self._exit_direction,
        }

        return data

    def classify(self, classifier: BaseModel | None = None):
        """compute the class estimate and confidence for this vehicle instance

        If no classifier is specified, the estimate is based on the argmax
        of the averages of all class probabilities given for the vehicle so far.

        If a classifier is specified, then the detector estimates are ignored,
        and an ensemble-like algorithm is employed using soft-voting on the
        average probability estimates per class over a specific number of frames
        from this vehicles lifespan.

        NOTE: the class and class confidence attributes are set after calling
        this function so that if .get_data() is called afterwards they will be
        included
        """
        if classifier is None:
            # single stage, just get max estimate
            cls_num = np.argmax(list(map(float, self._detector_class_conf_bins)))
            conf = float(self._detector_class_conf_bins[cls_num])

            self._class = self._detector_class_conf_bins.get(cls_num)
            self._class_conf = conf
        else:
            # two-stage, so run classifier in ensemble-like fashion
            class_bins = np.zeros(
                (len(self._detections), len(classifier.get_classes()))
            )

            for idx, det in enumerate(self._detections):
                # slice out single object from image, making sure to clip coords
                # outside of the image boundaries
                bbox = det.bbox.to_int()
                bbox.clip(det.img.shape[1], det.img.shape[0])
                x1, y1 = bbox.top_left.as_tuple()
                x2, y2 = bbox.bottom_right.as_tuple()

                img_slice = det.img[y1:y2, x1:x2]
                result = classifier.inference(img_slice)
                class_bins[idx, :] = result

            print(class_bins)
            classifier_class_map = classifier.get_classes()
            cls_estimates = np.mean(class_bins, axis=0)
            cls_num = np.argmax(cls_estimates)
            conf = cls_estimates[cls_num]

            self._class = classifier_class_map.get(cls_num)
            self._class_conf = conf

    def compute_directions(
        self,
        zone_mask: np.ndarray | None,
        zone_mask_map: Dict[int, str] | None,
        img_w: int,
        img_h: int,
    ):
        """computes the entry and exit directions based on the vehicle instances
        entry coordinate and it's current coordinate by fitting a line between
        them and computing the intersection of those lines and the boundaries of
        the image

        NOTE: sets the class attributes for entry and exit directions so that
        if .get_data() is called afterwards the directions will be included
        """
        # only one point, so cannot compute a line
        if self._entry_coord == self._current_coord:
            return

        # no zone_mask, can't compute directions
        if zone_mask is None or zone_mask_map is None:
            return

        # Fit a straight line (y = mx + b) to the two points
        if (
            self._current_coord.x != self._entry_coord.x
            and self._current_coord.y != self._entry_coord.y
        ):
            # not vertical or horizontal, proceed with normal calcs
            m = (self._current_coord.y - self._entry_coord.y) / (
                self._current_coord.x - self._entry_coord.x
            )
            b = self._entry_coord.y - m * self._entry_coord.x
            is_vertical, is_horizontal = False, False
        elif self._current_coord.y == self._entry_coord.y:
            # horizontal line
            is_vertical, is_horizontal = False, True
        else:
            # vertical line
            is_vertical, is_horizontal = True, False

        # Calculate intersections with image boundaries. Associate points to intersections
        # by computing distance from one intersection to both points, and then associate
        # the closest point to that intersection and the other point to the other intersection
        if is_vertical:
            # print("Line was vertical")
            # top of image
            intersection1 = np.array((self._entry_coord.x, 0)).astype(int)
            # bottom of image
            intersection2 = np.array((self._entry_coord.x, img_h)).astype(int)
        elif is_horizontal:
            # print("Line was horizontal")
            # left of image
            intersection1 = np.array((0, self._entry_coord.y)).astype(int)
            # right of image
            intersection2 = np.array((img_w, self._entry_coord.y)).astype(int)
        else:
            # print("Line was diagonal")
            intersections = []
            # top of image
            intersections.append(np.array((-b / m, 0)).astype(int))
            # bottom of image
            intersections.append(np.array(((img_h - b) / m, img_h)).astype(int))
            # left of image
            intersections.append(np.array((0, b)).astype(int))
            # right of image
            intersections.append(np.array((img_w, img_w * m + b)).astype(int))

            valid_intersections = []
            for intersect in intersections:
                if len(valid_intersections) == 2:
                    # edge case where line becomes perfect diagonal across the image,
                    # causes duplicate intersection points
                    break
                if not intersect[0] >= 0 or not intersect[0] <= img_w:
                    # not a valid x-coord
                    continue
                if not intersect[1] >= 0 or not intersect[1] <= img_h:
                    # not a valid y-coord
                    continue
                valid_intersections.append(intersect)

            if len(valid_intersections) != 2:
                # something funky happened and we can't compute zones
                return

            intersection1 = valid_intersections[0]
            intersection2 = valid_intersections[1]

        entry_inter1_dist = np.linalg.norm(intersection1 - self._entry_coord.as_numpy())
        exit_inter1_dist = np.linalg.norm(
            intersection1 - self._current_coord.as_numpy()
        )

        intersection1_dir = zone_mask_map[zone_mask[intersection1[1], intersection1[0]]]
        intersection2_dir = zone_mask_map[zone_mask[intersection2[1], intersection2[0]]]

        if entry_inter1_dist < exit_inter1_dist:
            # entry point closer to intersection1 than exit point
            self._entry_direction = intersection1_dir
            self._exit_direction = intersection2_dir
        else:
            # exit point closer to intersection1 than entry point
            self._entry_direction = intersection2_dir
            self._exit_direction = intersection1_dir

    def update_coords(self, coordinate: Point):
        """updates the current centroid estimate of the vehicle, and if it hasn't
        been set before then it get's set as the vehicles entry coordinate"""
        if self._entry_coord is None:
            self._entry_coord = coordinate
        self._current_coord = coordinate

    def update_class_estimate(self, class_num: int, class_conf: float):
        """updates the class estimate and confidence bins based on model inference"""
        self._detector_class_conf_bins[class_num].update(class_conf)

    def update_detections(self, detections: List[Detection]):
        """updates the detections stored for this vehicle. Should be relatively
        uniformly distributed across its lifespan (according to norfair library)
        """
        updated_dets = []
        for det in detections:
            updated_dets.append(
                VehicleDetection(
                    img=det.data.get("img"),
                    bbox=points_to_rect(det.points),
                    cls=det.data.get("class"),
                    conf=det.data.get("conf"),
                    frame_idx=det.data.get("frame_idx"),
                )
            )
        # Just overwriting for now, may change in the future
        self._detections = updated_dets

    def increment_frame_count(self):
        """increments the number of frames this vehicle has been alive for"""
        self._num_of_frames += 1
