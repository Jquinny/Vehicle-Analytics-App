from typing import List, Tuple, Dict

import cv2 as cv
import norfair
import numpy as np

from src.utils.geometry import Point, Poly


def _merge_frames(track_mask: np.ndarray, video_frame: np.ndarray):
    """combine the mask created by the norfair path drawer and the video frame
    in order to draw correct tracklets
    """
    return cv.addWeighted(track_mask, 1, video_frame, 1, 0)


def draw_zones(img: np.ndarray, zone_mask: np.ndarray):
    """draws the zone lines on the image"""
    edges = cv.Canny((zone_mask * 50).astype(np.uint8), threshold1=100, threshold2=200)
    img[edges == 255] = (0, 0, 255)


def draw_roi(img: np.ndarray, roi: Poly, close: bool = False):
    """Draw ROI polygon onto image"""
    if roi:
        for pt in roi:
            cv.circle(img, pt.to_int().as_tuple(), 3, (0, 255, 0), 3)
        cv.polylines(
            img,
            [np.array([pt.as_tuple() for pt in roi.to_int()])],
            close,
            (0, 255, 0),
            2,
        )


def draw_coordinates(
    frame: np.ndarray, coordinates: List[Point], coord_map: Dict[int, str]
):
    if coordinates:
        for idx, pt in enumerate(coordinates):
            text = coord_map[idx]
            center = pt.to_int().as_tuple()
            radius = 5
            cv.circle(frame, center, radius, (0, 0, 255), -1)

            text_color = (255, 255, 255)  # White text
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_size = cv.getTextSize(text, font, font_scale, 1)[0]

            rect_size = (text_size[0] + 10, text_size[1] + 10)  # Square around the text
            rect_top_left = (
                center[0] - rect_size[0] // 2,
                center[1] - radius - rect_size[1],
            )
            rect_bottom_right = (center[0] + rect_size[0] // 2, center[1] - radius)
            rect_background_color = (0, 0, 0)  # Black background
            cv.rectangle(
                frame, rect_top_left, rect_bottom_right, rect_background_color, -1
            )

            text_x = center[0] - text_size[0] // 2
            text_y = (
                center[1] - radius - 5
            )  # Adjust to place text just above the circle
            cv.putText(
                frame,
                text,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                1,
                cv.LINE_AA,
            )


# NOTE: no clue if this is actually right but it kind of looks like it may be
def draw_vector(
    img: np.ndarray,
    start_pt: Tuple[int, int],
    direction_vec: Tuple[float, float],
    vec_len: int = 100,
):
    """helper for drawing vectors from a starting point in the direction specified
    in direction_vec.
    """
    theta = np.arctan2(direction_vec[1], direction_vec[0])
    x = int(vec_len * np.cos(theta) + start_pt[0])
    y = int(vec_len * np.sin(theta) + start_pt[1])
    cv.arrowedLine(img, start_pt, (x, y), (0, 255, 0))


def draw_tracker_predictions(
    frame: np.ndarray,
    tracked_objects: List[norfair.tracker.TrackedObject],
    track_points: str = "bbox",
    draw_ids: bool = True,
    thickness: int = 2,
):
    """helper for drawing the tracker predictions"""

    if track_points == "bbox":
        norfair.draw_boxes(
            frame, tracked_objects, draw_ids=draw_ids, thickness=thickness
        )
    elif track_points == "centroid":
        norfair.draw_points(
            frame, tracked_objects, draw_ids=draw_ids, thickness=thickness
        )


def draw_detector_predictions(
    frame: np.ndarray,
    detections: List[norfair.tracker.Detection],
    track_points: str = "bbox",
    draw_labels: bool = True,
    draw_scores: bool = True,
    thickness: int = 2,
):
    """helper for drawing the detector predictions"""

    if track_points == "bbox":
        norfair.draw_boxes(
            frame,
            detections,
            draw_labels=draw_labels,
            draw_scores=draw_scores,
            thickness=thickness,
        )
    elif track_points == "centroid":
        norfair.draw_points(
            frame,
            detections,
            draw_labels=draw_labels,
            draw_scores=draw_scores,
            thickness=thickness,
        )


def draw_tracklets(
    frame: np.ndarray,
    path_drawer: norfair.Paths | None,
    tracked_objects: List[norfair.tracker.TrackedObject],
):
    """helper for drawing tracklet paths associated with objects"""

    if len(tracked_objects) == 0:
        tracklet_frame = np.zeros_like(frame, dtype=np.uint8)
    else:
        tracklet_frame = path_drawer.draw(frame, tracked_objects)

    frame = _merge_frames(tracklet_frame, frame)
