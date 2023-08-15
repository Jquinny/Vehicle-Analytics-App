from typing import List, Dict, Tuple

import cv2 as cv
import numpy as np

from src.utils.geometry import Point, Poly
from src.utils.drawing import draw_coordinates, draw_roi

WINDOW_SIZE = (1280, 720)


def get_roi_points(event, x, y, flags, data: dict):
    """Get list of points for ROI from user mouse input"""
    if event == cv.EVENT_LBUTTONDOWN:
        data["roi"].push(Point(x=x, y=y))

    if event == cv.EVENT_RBUTTONDOWN:
        if data["roi"]:
            data["roi"].pop()
            data["copy"] = data["original"].copy()  # erasing lines to redraw

    draw_roi(data["copy"], data["roi"])

    return


def get_roi(frame: np.ndarray) -> Poly | None:
    window_title = "Region of Interest Extraction"
    window_scale_x = frame.shape[1] / WINDOW_SIZE[0]
    window_scale_y = frame.shape[0] / WINDOW_SIZE[1]
    resized_frame = cv.resize(frame, WINDOW_SIZE)
    data = {"roi": Poly(), "original": resized_frame, "copy": resized_frame.copy()}

    # grab ROI from user
    cv.namedWindow(window_title)
    cv.setMouseCallback(window_title, get_roi_points, data)
    print("\nPress enter to connect final points.")
    print("Press enter again if satisfied. Press any other key to continue editing.\n")
    while True:
        cv.imshow(window_title, data["copy"])
        key = cv.waitKey(10)
        if key == ord("\r"):  # show full polygon when user presses enter
            data["copy"] = data["original"].copy()
            draw_roi(data["copy"], data["roi"], True)
            cv.imshow(window_title, data["copy"])
            key = cv.waitKey(0)
            if key == ord("\r"):
                cv.destroyWindow(window_title)
                break

    if not data["roi"].is_empty():
        # coordinate transform due to window scaling
        return Poly(
            [
                Point(x=pt.x * window_scale_x, y=pt.y * window_scale_y)
                for pt in data["roi"]
            ]
        )
    else:
        return None


def get_coordinate_points(event, x, y, flags, data: dict):
    """Get list of points for ROI from user mouse input"""
    if event == cv.EVENT_LBUTTONDOWN:
        if len(data["coordinates"]) < 4:
            data["coordinates"].append(Point(x=x, y=y))

    if event == cv.EVENT_RBUTTONDOWN:
        if data["coordinates"]:
            data["coordinates"].pop()
            data["copy"] = data["original"].copy()  # erasing coords to redraw

    draw_coordinates(data["copy"], data["coordinates"], data["coord_map"])

    return


def get_coordinates(frame: np.ndarray) -> Tuple[List[Point], Dict[int, str]]:
    window_title = "Direction Coordinate Placement"
    window_scale_x = frame.shape[1] / WINDOW_SIZE[0]
    window_scale_y = frame.shape[0] / WINDOW_SIZE[1]
    resized_frame = cv.resize(frame, WINDOW_SIZE)
    data = {
        "coordinates": [],
        "coord_map": {0: "N", 1: "S", 2: "E", 3: "W"},
        "original": resized_frame,
        "copy": resized_frame.copy(),
    }

    # grab direction coordinates from user
    cv.namedWindow(window_title)
    cv.setMouseCallback(window_title, get_coordinate_points, data)
    print("\nPress enter to confirm.")
    while True:
        cv.imshow(window_title, data["copy"])
        key = cv.waitKey(10)
        if len(data["coordinates"]) == 4 and key == ord("\r"):
            cv.destroyWindow(window_title)
            break

    if data["coordinates"]:
        # coordinate transform due to window scaling
        data["coordinates"] = [
            Point(x=pt.x * window_scale_x, y=pt.y * window_scale_y)
            for pt in data["coordinates"]
        ]

    return data["coordinates"], data["coord_map"]
