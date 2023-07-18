"""Main script for running the full truck analysis pipeline.

TODO:
- add either ROI input as well as NSEW directions or just NSEW directions
- implement data aggregation through lifecycle of vehicle
- check out position and velocity estimate properties of the TrackedObject class
for estimating entry and exit positions as well as the speed (velocity might
be an issue because the property is in absolute coordinates)
- implement a ReID function for when kalman filter estimates fail (check out the
data parameter of Detection objects for this)
"""

import argparse
import os

import numpy as np
import torch

import cv2 as cv
import norfair
from norfair import Detection, Paths, Tracker, Video
from norfair.utils import print_objects_as_table  # for testing ............
from ultralytics import YOLO

from config import ROOT_DIR
from utils.tracking import yolo_detections_to_norfair_detections, merge_frames
from utils.user_input import get_roi, draw_roi, get_coordinates, ROI, COORDINATES

# NOTE: play with these constants, or try an algorithm that somehow finds the
# best results once I have a ground truth set up

# tracking constants
DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000
LIFESPAN = 15  # max number of frames a tracked object can surive without a match
DETECTION_THRESHOLD = 0  # may be redundant if I already have CONF_THRESHOLD

# detection constants
IMG_SZ = 512
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45


def process(model_path, video_path, track_points="bbox"):
    # automatically set device for inferencing
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"using device {device}")

    model = YOLO(model_path, task="detect")

    video = Video(input_path=video_path)

    distance_function = "iou" if track_points == "bbox" else "euclidean"
    distance_threshold = (
        DISTANCE_THRESHOLD_BBOX
        if track_points == "bbox"
        else DISTANCE_THRESHOLD_CENTROID
    )

    tracker = Tracker(
        distance_function=distance_function,
        distance_threshold=distance_threshold,
    )

    path_drawer = Paths(attenuation=0.05)

    for frame_num, frame in enumerate(video):
        # grab info on first frame
        if frame_num == 0:
            # these will be stored in ROI and COORDINATES afterwards
            get_roi(frame)
            get_coordinates(frame)

        # get detections from model inference
        yolo_detections = model.predict(
            frame,
            imgsz=IMG_SZ,
            device=device,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        # convert to format norfair can use
        detections = yolo_detections_to_norfair_detections(
            yolo_detections, track_points=track_points
        )

        # update tracker with new detections
        tracked_objects = tracker.update(detections=detections)
        print_objects_as_table(tracked_objects)  # for testing ..............

        # NOTE: check if I should be doing this
        if len(tracked_objects) > 0:
            # TODO: update vehicle states using VehicleInstanceTracker
            pass

        # drawing stuff
        if track_points == "centroid":
            norfair.draw_points(frame, tracked_objects, thickness=2)
            norfair.draw_points(
                frame, detections, draw_labels=True, draw_scores=True, thickness=2
            )
            tracklet_frame = path_drawer.draw(frame, tracked_objects)
        elif track_points == "bbox":
            norfair.draw_boxes(frame, tracked_objects, thickness=2)
            norfair.draw_boxes(
                frame, detections, draw_labels=True, draw_scores=True, thickness=2
            )
            tracklet_frame = path_drawer.draw(frame, tracked_objects)

        if len(tracked_objects) == 0:
            tracklet_frame = np.zeros_like(frame, dtype=np.uint8)
        output_frame = merge_frames(tracklet_frame, frame)

        # video.write(output_frame)
        cv.imshow("output", output_frame)
        if cv.waitKey(10) == ord("q"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track objects in a video.")
    parser.add_argument(
        "--track-points",
        type=str,
        default="bbox",
        help="Track points: 'centroid' or 'bbox'",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="model weight filename in form <filename>.pt",
        default="small_training.pt",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="video filename in form <filename>.<ext>",
        default="test_videos/short_video.mp4",
    )
    args = parser.parse_args()
    model_path = args.model
    video_path = args.video
    track_points = args.track_points

    process(model_path, video_path, track_points)
