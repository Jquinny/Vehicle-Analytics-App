"""Main script for running the full truck analysis pipeline.

TODO:
- add either ROI input as well as NSEW directions or just NSEW directions
- implement data aggregation through lifecycle of vehicle
- check out position and velocity estimate properties of the TrackedObject class
for estimating entry and exit positions as well as the speed (velocity might
be an issue because the property is in absolute coordinates)
- add ability to select detector model TYPE (yolov8, rt-detr, efficientdet, etc.)
- implement a ReID function for when kalman filter estimates fail (check out the
data parameter of Detection objects for this)
"""

import argparse
import os
import random
import time

import numpy as np
import torch

import cv2 as cv
import easyocr
import norfair
from norfair import Paths, Tracker, Video
from norfair.utils import print_objects_as_table  # for testing ............
from ultralytics import YOLO

from config import ROOT_DIR, CLASSIFIER_NAME2NUM
from src.tracking import (
    yolo_detections_to_norfair_detections,
    merge_frames,
    VehicleInstanceTracker,
)
from utils.user_input import get_roi, draw_roi, get_coordinates, ROI, COORDINATES
from utils.drawing import (
    draw_detector_predictions,
    draw_tracker_predictions,
    draw_tracklets,
)
from utils.image import parse_timestamp
from utils.video import VideoHandler

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


def process(
    model_path,
    video_path,
    track_points="bbox",
    show_detector_predictions=True,
    show_tracker_predictions=True,
    show_tracklets=True,
    save_video=False,
):
    # automatically set device for inferencing
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"using device {device}")

    # TODO: use model factories here for classifier and detector. Should take in
    # argument that says what detection model and what classifier model as well
    # as weights
    model = YOLO(model_path, task="detect")

    # setup ocr model
    reader = easyocr.Reader(["en"])

    # setting up the video for reading/writing frames
    video_output_dir = str(ROOT_DIR / "output_videos")
    video_handler = VideoHandler(input_path=video_path, output_dir=video_output_dir)

    # setting the distance parameters for the tracker
    distance_function = "iou" if track_points == "bbox" else "euclidean"
    distance_threshold = (
        DISTANCE_THRESHOLD_BBOX
        if track_points == "bbox"
        else DISTANCE_THRESHOLD_CENTROID
    )

    # set up vehicle tracker
    vehicle_tracker = VehicleInstanceTracker(
        distance_function=distance_function, distance_threshold=distance_threshold
    )

    # utility for drawing tracklets
    path_drawer = Paths(attenuation=0.05)

    # grab roi, direction coordinates, and timestamp
    frame = video_handler.get_frame(0)
    roi = get_roi(frame)
    direction_coords = get_coordinates(frame)
    initial_datetime = parse_timestamp(frame, reader)
    print(roi)
    print(direction_coords)
    print(initial_datetime)

    for frame_num, frame in enumerate(video_handler):
        print(video_handler.current_frame)
        print(video_handler.get_time_from_frame(video_handler.current_frame))

        # get detections from model inference
        detections = model.predict(
            frame,
            imgsz=IMG_SZ,
            device=device,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        # convert to format norfair can use
        norfair_detections = yolo_detections_to_norfair_detections(
            detections, track_points=track_points
        )

        # TODO: do classification before tracking I think. This way we can add
        # the class label and the confidence to the detection data for vehicle
        # state updating NOTE: have to do it afterwards because the class label
        # dictates if the tracker matches or not

        # update tracker with new detections
        tracked_objects = vehicle_tracker.update(frame, norfair_detections)

        # drawing stuff
        if show_detector_predictions:
            draw_detector_predictions(frame, norfair_detections, track_points)
        if show_tracker_predictions:
            draw_tracker_predictions(frame, tracked_objects, track_points)
        if show_tracklets:
            draw_tracklets(frame, path_drawer, tracked_objects)

        if save_video:
            video_handler.write_frame_to_video(frame)

        if video_handler.show(frame, 10) == ord("q"):
            video_handler.cleanup()
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
    parser.add_argument(
        "--show_detector_predictions",
        help="show detector predictions on video frames",
        action="store_true",
    )
    parser.add_argument(
        "--show_tracker_predictions",
        help="show tracker predictions on video frames",
        action="store_true",
    )
    parser.add_argument(
        "--show_tracklets",
        help="show object tracklet paths on video frames",
        action="store_true",
    )
    parser.add_argument(
        "--save_video", help="write video to a file", action="store_true"
    )
    args = parser.parse_args()
    model_path = args.model
    video_path = args.video
    track_points = args.track_points
    show_detector_predictions = args.show_detector_predictions
    show_tracker_predictions = args.show_tracker_predictions
    show_tracklets = args.show_tracklets
    save_video = args.save_video

    process(
        model_path,
        video_path,
        track_points,
        show_detector_predictions,
        show_tracker_predictions,
        show_tracklets,
        save_video,
    )
