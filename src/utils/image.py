from typing import Tuple, List
import datetime
import re

import cv2 as cv
import easyocr
import numpy as np

from norfair import Detection
from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)
from src.utils.geometry import Rect, points_to_rect
from src.utils.video import VideoHandler


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
        0: r"\d{4}.\d{2}.\d{2}.\d{2}.\d{2}.\d{2}.\d{4}",
        1: r"\d{4}.\d{2}.\d{2}.\d{1,2}.\d{2}.\d{2}.[aApP][mM]",
    }

    def str_convert(detected_text: str) -> datetime.datetime | None:
        """converts a string into a datetime object"""
        if re.match(formats[0], detected_text):
            info_list = re.split(r"\D+", detected_text)
            if len(info_list) < 6:
                # error parsing the timestamp, just return None
                return None

            if len(info_list[2]) == 5:
                # silly case where ocr reader sees T as a digit
                silly_string = info_list[2]
                info_list[2] = silly_string[:2]
                info_list.insert(3, silly_string[3:])

            date = "-".join(info_list[:3])
            time = ":".join(info_list[3:6])

            datetime_str = " ".join([date, time])
            try:
                dt = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
            except:
                dt = None
            return dt
        elif re.match(formats[1], detected_text):
            info_list = re.split(r"\D+", detected_text)
            if len(info_list) < 6:
                # error parsing the timestamp, just return None
                return None

            date = "-".join(info_list[:3])
            time = ":".join(info_list[3:6])

            if detected_text[-2:] == "PM":
                # convert to 24hr format
                time_components = time.split(":")
                time_components[0] = str(12 + int(time_components[0]))
                time = ":".join(time_components)

            datetime_str = " ".join([date, time])
            try:
                dt = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
            except:
                dt = None
            return dt
        else:
            return None

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(img_gray, 235, 255, cv.THRESH_BINARY)

    text_detections = reader.readtext(thresh, detail=0)
    for text in text_detections:
        datetime_str = str_convert(text)
        if datetime_str:
            return datetime_str

    return None


# helper function for slicing single object images out of detection objects
def extract_objects(detections: List[Detection], img: np.ndarray) -> List[np.ndarray]:
    """helper for extracting single object images from bounding boxes

    NOTE: the detections should all be for the same image

    Arguments
    ---------
    detections (List[Detection]):
        the list of norfair detection objects holding the necessary bbox and
        frame info
    img (np.ndarray):
        the image that the objects should be extracted from

    Returns
    -------
    List[np.ndarray]:
        the list of single object images, in the same order that the detection
        objects were in
    """
    img_list = []
    for det in detections:
        # slice out single object from image, making sure to clip coords
        # outside of the image boundaries
        bbox = points_to_rect(det.points).to_int()
        bbox.clip(img.shape[1], img.shape[0])
        x1, y1 = bbox.top_left.as_tuple()
        x2, y2 = bbox.bottom_right.as_tuple()
        img_slice = img[y1:y2, x1:x2]
        img_list.append(img_slice)
    return img_list


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
    img: np.ndarray, rect: Rect, area_pct: float = 0.20
) -> Tuple[int, int, int]:
    """computes the average color within a bounding box by computing the
    average color in a square near the center of the bounding box
    (since the road overtakes the color value in most bounding boxes)
    """
    if area_pct > 1.0:
        area_pct = 1.0
    elif area_pct < 0.0:  # NOTE: check this
        area_pct = 0.0

    center_x, center_y = rect.center.to_int().as_tuple()

    new_x1 = int(center_x - area_pct * rect.width // 2)
    new_y1 = int(center_y - area_pct * rect.height // 2)
    new_x2 = int(center_x + area_pct * rect.width // 2)
    new_y2 = int(center_y + area_pct * rect.height // 2)

    # using opencv for images so bgr instead of rgb
    b = int(np.mean(img[new_y1:new_y2, new_x1:new_x2, 0]))
    g = int(np.mean(img[new_y1:new_y2, new_x1:new_x2, 1]))
    r = int(np.mean(img[new_y1:new_y2, new_x1:new_x2, 2]))

    return (r, g, b)
