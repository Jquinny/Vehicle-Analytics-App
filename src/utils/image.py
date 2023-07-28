from typing import Tuple, List
import datetime
import re

import cv2 as cv
import easyocr
import numpy as np


def bbox_center(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    """computes the center of a bounding box using top left and bottom right
    corner coordinates.

    Arguments
    ---------
    img (np.ndarray):
        the image being processed
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
        """converts a string into a datetime"""
        if re.match(formats[0], detected_text):
            split_timestamp = detected_text.split("T")
            date = split_timestamp[0]
            time = split_timestamp[1].split("-")[0]
            return " ".join([date, time])
        elif re.match(formats[1], detected_text):
            split_timestamp = detected_text.split(" ")
            date = split_timestamp[0]
            time = split_timestamp[1]

            if split_timestamp[2] == "PM":
                # convert to 24hr format
                time_components = time.split(":")
                time_components[0] = str(12 + int(time_components[0]))
                time = ":".join(time_components)

            return " ".join([date, time])
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


def check_overlap(
    region1: List[Tuple[int, int]], region2: List[Tuple[int, int]], pct: float = 0.30
):
    # TODO: implement the overlap check for bounding boxes and region of interest
    pass
