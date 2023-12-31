from typing import List, Tuple, Dict
import math

import cv2 as cv
import matplotlib.pyplot as plt
import norfair
import numpy as np

from src.utils.geometry import Point, Poly, Rect


def _merge_frames(track_mask: np.ndarray, video_frame: np.ndarray):
    """combine the mask created by the norfair path drawer and the video frame
    in order to draw correct tracklets
    """
    return cv.addWeighted(track_mask, 1, video_frame, 1, 0)


def draw_text(
    img: np.ndarray,
    text: str,
    top_left: Point,
    font_color: Tuple[int, int, int] = (255, 255, 255),
    background_color: Tuple[int, int, int] = (0, 0, 0),
):
    """draws white text over a black rectangular background, with the background
    being placed with it's top left corner at the point top_left specifies
    """
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5

    text_size = cv.getTextSize(text, font, font_scale, 1)[0]
    text_width, text_height = text_size

    # Calculate the position of the background rectangle
    rect_top_left = top_left.to_int().as_tuple()
    rect_bottom_right = (
        rect_top_left[0] + text_width + 5,
        rect_top_left[1] + text_height + 5,
    )

    # Draw the background rectangle
    cv.rectangle(img, rect_top_left, rect_bottom_right, background_color, -1)

    # Calculate the position of the text (origin of text is bottom left corner)
    text_position = (
        rect_top_left[0] + 2,
        rect_bottom_right[1] - 2,
    )  # Slightly adjust the text position

    # Draw the text
    cv.putText(img, text, text_position, font, font_scale, font_color, 1, cv.LINE_AA)


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


def draw_coordinates(frame: np.ndarray, coordinates: Dict[str, Point]):
    """helper for drawing direction coordinates onto the image frame"""
    for text, pt in coordinates.items():
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
        cv.rectangle(frame, rect_top_left, rect_bottom_right, rect_background_color, -1)

        text_x = center[0] - text_size[0] // 2
        text_y = center[1] - radius - 5  # Adjust to place text just above the circle
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


def draw_rect(
    img: np.ndarray,
    rect: Rect,
    tracker_id: str = None,
    class_name: str = None,
    conf: str = None,
    color: Tuple[int, int, int] | None = None,
):
    """helper function for drawing bounding boxes on an image"""

    top_left = rect.top_left.to_int().as_tuple()
    bottom_right = rect.bottom_right.to_int().as_tuple()

    # assign color if none assigned
    if color is None:
        color = (255, 0, 0)  # defaults to blue

    cv.rectangle(
        img,
        top_left,
        bottom_right,
        color,
        3,
    )

    if tracker_id or class_name or conf:
        if not tracker_id:
            tracker_id = ""
        if not class_name:
            class_name = ""
        if not conf:
            conf = ""

        text = f"{tracker_id} {class_name} {conf:.2f}"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]

        text_background_tl = (top_left[0], top_left[1] - text_size[1] - 5)
        text_background_br = (top_left[0] + text_size[0], top_left[1] - 5)

        cv.rectangle(img, text_background_tl, text_background_br, (255, 255, 255), -1)
        cv.putText(
            img,
            text,
            (top_left[0], top_left[1] - 5),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            lineType=cv.LINE_AA,
        )


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


def plot_images(images: List[np.ndarray], titles: List[str] | None = None):
    """helper function for plotting multiple images in a subplot."""

    if titles is not None:
        assert len(images) == len(titles), "# of titles does not match # of images"

    nrows = math.ceil(math.sqrt(len(images)))
    fig, axes = plt.subplots(nrows=nrows, ncols=nrows)  # square grid
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            ax.imshow(images[idx])
            if titles is not None:
                ax.set_title(titles[idx])
        ax.axis("off")
    plt.show()
