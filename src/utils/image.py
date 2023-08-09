from typing import Tuple, List
import datetime
import re

import cv2 as cv
import easyocr
import numpy as np

from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)


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


def parse_timestamp(img: np.ndarray, reader: easyocr.Reader) -> str | None:
    """finds the timestamp in a video. Assumes the timestamp is on a black bar

    Arguments
    ---------
    img (np.ndarray):
        the image to find the timestamp in
    reader (easyocr.Reader):
        the ocr reader

    Returns
    -------
    str | None:
        the timestamp found in the image as a string formatted in the following
        way: "year-month-day hour:min:sec". Time is in 24hr clock format.
        If no timestamp could be parsed, returns None
    """

    formats = {
        0: r"\d{4}-\d{2}-\d{2}T\d{2}\:\d{2}\:\d{2}-\d{4}",
        1: r"\d{4}-\d{2}-\d{2} \d{1,2}\:\d{2}\:\d{2} (AM|PM)",
    }

    def str_convert(detected_text: str) -> datetime.datetime | None:
        """converts a string into a datetime object"""
        if re.match(formats[0], detected_text):
            split_timestamp = detected_text.split("T")
            date = split_timestamp[0]
            time = split_timestamp[1].split("-")[0]
            datetime_str = " ".join([date, time])
            return datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        elif re.match(formats[1], detected_text):
            split_timestamp = detected_text.split(" ")
            date = split_timestamp[0]
            time = split_timestamp[1]

            if split_timestamp[2] == "PM":
                # convert to 24hr format
                time_components = time.split(":")
                time_components[0] = str(12 + int(time_components[0]))
                time = ":".join(time_components)

            datetime_str = " ".join([date, time])
            return datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        else:
            return None

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(img_gray, 240, 255, cv.THRESH_BINARY)

    text_detections = reader.readtext(thresh, detail=0)
    for text in text_detections:
        text = text.replace(".", ":")
        datetime_str = str_convert(text)
        if datetime_str:
            return datetime_str

    return None


def get_color_from_RGB(rgb_tuple: Tuple[int, int, int]) -> str:
    """returns the human readable color name from the RGB color tuple associated
    with this vehicle

    Arguments
    ---------
    rgb_tuple (Tuple[int, int, int]):
        the color in the rgb tuple format

    Returns
    -------
    str:
        the human readable color name
    """

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


def compute_avg_color(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int, area_pct: float = 0.20
) -> Tuple[int, int, int]:
    """computes the average color within a bounding box by computing the
    average color in a square near the center of the bounding box
    (since the road overtakes the color value in most bounding boxes)
    """
    if area_pct > 1.0:
        area_pct = 1.0
    elif area_pct < 0.0:  # NOTE: check this
        area_pct = 0.0

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


def check_overlap(
    region1: List[Tuple[int, int]], region2: List[Tuple[int, int]], pct: float = 0.30
):
    # TODO: implement the overlap check for bounding boxes and region of interest
    pass
