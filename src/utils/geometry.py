from __future__ import annotations

from typing import Tuple, List
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from shapely.geometry import Polygon


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
