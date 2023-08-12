from __future__ import annotations

from typing import Tuple, List
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from shapely.geometry import Polygon


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
    """TODO:
    -   make this implementation a wrapper around a list of Point objects that
        can create shapely polygons, do validity checks, and add/remove points

    NEED TO FIGURE OUT IF I WANT TO DO THE VALIDITY CHECK AS CLASS METHOD OR
    PART OF THE COORDS LIST WRAPPER IMPLEMENTATION
    """

    def __init__(self, coords: List[Point] | None):
        self.coords = coords

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
        print(self)
        print(polygon)
        p1 = self.as_shapely()
        p2 = polygon.as_shapely()
        area_overlap = p1.intersection(p2).area / p2.area
        print(area_overlap)

        return area_overlap > overlap_pct


# NOTE: DELETE WHATS BELOW


def bbox_center(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    """computes the center of a bounding box using top left and bottom right
    corner coordinates.

    Arguments
    ---------
    x1 (int):
        top left bounding box corner x-coord
    y1 (int):
        top left bounding box corner y-coord
    x2 (int):
        bottom right bounding box corner x-coord
    y2 (int):
        bottom right bounding box corner y-coord

    Returns
    -------
    int:
        x coord of the center of the bounding box
    int:
        y coord of the center of the bounding box
    """

    center_x = int(x1 + (x2 - x1) // 2)
    center_y = int(y1 + (y2 - y1) // 2)

    return center_x, center_y


def bbox_area(x1: int, y1: int, x2: int, y2: int) -> int:
    """computes the area of a bounding box using top left and bottom right
    corner coordinates.

    Arguments
    ---------
    x1 (int):
        top left bounding box corner x-coord
    y1 (int):
        top left bounding box corner y-coord
    x2 (int):
        bottom right bounding box corner x-coord
    y2 (int):
        bottom right bounding box corner y-coord

    Returns
    -------
    int:
        bounding box area in terms of number of pixels
    """

    return int(x2 - x1) * int(y2 - y1)


def bbox_to_polygon(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
    """creates a polygon representing a bounding box based on the

    Arguments
    ---------
    x1 (int):
        top left bounding box corner x-coord
    y1 (int):
        top left bounding box corner y-coord
    x2 (int):
        bottom right bounding box corner x-coord
    y2 (int):
        bottom right bounding box corner y-coord

    Returns
    -------
    List[Tuple[int, int]]
    """
    top_left = (x1, y1)
    top_right = (x2, y1)
    bottom_left = (x1, y2)
    bottom_right = (x2, y2)

    return [top_left, top_right, bottom_right, bottom_left]


def check_overlap(region1: np.ndarray, region2: np.ndarray, overlap_pct: float = 0.30):
    """Checks the overlap area of two polygons to see if it's greater than
    the required overlap percent.

    The overlap area percentage is computed with respect to the area of region2,
    so if you want the overlap to have an area > 50% of the area of region2
    then overlap_pct would be .50

    Arguments
    ---------
    region1 (np.ndarray):
        A polygon represented by a numpy array of shape (N, 2), containing the
        x, y coordinates of the points.top left bounding box corner x-coord
    region2 (np.ndarray):
        A polygon represented by a numpy array of shape (N, 2), containing the
        x, y coordinates of the points.top left bounding box corner x-coord
    overlap_pct (float):
        the percentage of overlap to check for

    Returns
    -------
    bool:
        whether or not the overlap area with respect to region2 was greater
        than the overlap_pct
    """
    print(region1)
    print(region2)
    p1 = Polygon(region1)
    p2 = Polygon(region2)
    area_overlap = p1.intersection(p2).area / p2.area
    print(area_overlap)

    return area_overlap > overlap_pct
