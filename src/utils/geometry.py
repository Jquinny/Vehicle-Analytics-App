from __future__ import annotations

from typing import Tuple, List
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from shapely.geometry import Polygon
from ultralytics.utils.ops import xyxy2xywhn


def points_to_rect(points: np.ndarray) -> Rect:
    """helper for converting 2x2 norfair.tracker.Detection.points attribute
    into a Rect object
    """
    x1, y1 = points[0]
    x2, y2 = points[1]
    return Rect(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


@dataclass
class Point:
    x: float
    y: float

    def to_int(self) -> Point:
        """convert attributes to integers"""
        return Point(x=int(self.x), y=int(self.y))

    def as_tuple(self) -> Tuple:
        return self.x, self.y

    def as_list(self) -> List:
        return [self.x, self.y]

    def as_numpy(self) -> np.array:
        return np.array([self.x, self.y])


@dataclass
class Rect:
    """x and y are assumed to be the top left corner of the rectangle"""

    x: float
    y: float
    width: float
    height: float

    def to_int(self) -> Rect:
        """convert attributes to integers"""
        return Rect(
            x=int(self.x), y=int(self.y), width=int(self.width), height=int(self.height)
        )

    def clip(self, img_w: int, img_h: int):
        """clips coordinates to ensure they are within a specific image shape"""
        bbox = self.to_numpy()
        bbox[[0, 2]] = bbox[[0, 2]].clip(0, img_w)
        bbox[[1, 3]] = bbox[[1, 3]].clip(0, img_h)

        self.x = bbox[0]
        self.y = bbox[1]
        self.width = bbox[2] - self.x if bbox[2] - self.x > 0 else 0
        self.height = bbox[3] - self.y if bbox[3] - self.y > 0 else 0

    def to_numpy(self) -> np.ndarray:
        """convert Rect to a 1D numpy array of form [x1 y1 x2 y2]"""
        return np.array([self.x, self.y, self.x + self.width, self.y + self.height])

    def to_yolo(self, img_w, img_h) -> Tuple[float, float, float, float]:
        """returns the rectangle information in the yolov8.txt format (i.e.
        [x y width height] where they are normalized with respect to the image)"""

        center_x = (self.x + (self.x + self.width)) / 2
        center_y = (self.y + (self.y + self.height)) / 2

        normalized_center_x = center_x / img_w
        normalized_center_y = center_y / img_h
        normalized_bbox_width = self.width / img_w
        normalized_bbox_height = self.height / img_h

        yolo_format = (
            normalized_center_x,
            normalized_center_y,
            normalized_bbox_width,
            normalized_bbox_height,
        )

        return yolo_format

    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def top_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y)

    @property
    def bottom_left(self) -> Point:
        return Point(x=self.x, y=self.y + self.height)

    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    @property
    def center(self) -> Point:
        center_x = self.x + self.width / 2
        center_y = self.y + self.height / 2
        return Point(x=center_x, y=center_y)

    @property
    def area(self) -> float:
        return self.width * self.height

    def pad(self, padding) -> Rect:
        return Rect(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2 * padding,
            height=self.height + 2 * padding,
        )

    def as_polygon(self) -> Poly:
        """returns the Rect as a Poly object with all 4 corners"""
        coords = [self.top_left, self.top_right, self.bottom_right, self.bottom_left]
        return Poly(coords)


class Poly:
    """Simple polygon class that allows pushing and popping Point objects, as
    well as checking polygon validity
    """

    def __init__(self, coords: List[Point] = []):
        self.coords = coords

    def __iter__(self):
        for pt in self.coords:
            yield pt

    def __len__(self):
        return len(self.coords)

    def push(self, pt: Point):
        self.coords.append(pt)

    def pop(self):
        if not self.is_empty():
            self.coords.pop()

    def to_int(self) -> Poly:
        int_coords = [pt.to_int() for pt in self.coords]
        return Poly(int_coords)

    def is_valid(self) -> bool:
        return self.as_shapely().is_valid

    def is_empty(self) -> bool:
        return len(self.coords) == 0

    def clear_coords(self):
        """clears all points in the polygon"""
        self.coords = []

    def as_shapely(self) -> Polygon:
        return Polygon([pt.as_tuple() for pt in self.coords])

    def check_overlap(self, polygon: Poly, overlap_pct: float = 0.15) -> bool:
        """Checks the overlap area of this polygon and the argument polygon to
        see if it's greater than the required overlap percent.

        The overlap area percentage is computed with respect to the area of user
        argument polygon, so if you want the overlap to have an area > 50% of
        the area of that polygon then overlap_pct would be .50

        Arguments
        ---------
        polygon (Poly):
            A polygon object to compute the overlap with respect to
        overlap_pct (float):
            the percentage of overlap to check for

        Returns
        -------
        bool:
            whether or not the overlap area with respect to polygon was greater
            than the overlap_pct
        """
        p1 = self.as_shapely()
        p2 = polygon.as_shapely()
        area_overlap = p1.intersection(p2).area / p2.area

        return area_overlap > overlap_pct
