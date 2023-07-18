from typing import List

import cv2 as cv
import norfair
import numpy as np


def _merge_frames(track_mask: np.ndarray, video_frame: np.ndarray):
    return cv.addWeighted(track_mask, 1, video_frame, 1, 0)


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
