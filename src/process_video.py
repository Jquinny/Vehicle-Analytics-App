"""Main script for running the full vehicle analysis pipeline.

TODO: setup so it works with new model_selector class
-   change interface so that the process() function takes in a task object,
    this task object will have all necessary info such as instantiated models
    and anything else required
"""

import argparse
import time

import numpy as np
import torch

import cv2 as cv
import easyocr
from norfair import Paths
from norfair.utils import print_objects_as_table  # for testing ............

from config import ROOT_DIR, CLASSIFIER_NAME2NUM
from src.tracking import VehicleInstanceTracker
from src.models.base_model import BaseModel
from src.utils.user_input import get_roi, draw_roi, get_coordinates, ROI, COORDINATES
from src.utils.drawing import (
    draw_polygon,
    draw_detector_predictions,
    draw_tracker_predictions,
    draw_tracklets,
)
from src.utils.geometry import check_overlap, bbox_to_polygon
from src.utils.image import parse_timestamp
from src.utils.video import VideoHandler
from src.model_selector import ModelSelector

# NOTE: play with these constants, or try an algorithm that somehow finds the
# best results once I have a ground truth set up

# tracking constants
DISTANCE_THRESHOLD_BBOX: float = 0.7
MAX_DISTANCE: int = 10000
LIFESPAN = 15  # max number of frames a tracked object can surive without a match
DETECTION_THRESHOLD = 0  # may be redundant if I already have CONF_THRESHOLD

# detection constants
IMG_SZ = 512
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45

ROI_OVERLAP_PCT = 0.15


def process(
    model: BaseModel,
    video_handler: VideoHandler,
    save_video: bool = False,
    debug: bool = False,
):
    """Main video processing function. accumulates vehicle data and writes out
    the video with bounding boxes, class names, and confidences, as well as
    a .json file with video metadata and a .csv file with all of the vehicle
    information.

    Arguments
    ---------
    """
    # automatically set device for inferencing
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if debug:
        print(f"using device {device}")

    # setup ocr model
    reader = easyocr.Reader(["en"])

    # setting the distance parameters for the tracker
    distance_function = "iou"
    distance_threshold = DISTANCE_THRESHOLD_BBOX

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

    # set up vehicle tracker
    vehicle_tracker = VehicleInstanceTracker(
        video_handler=video_handler,
        roi=roi,
        direction_coords=direction_coords,
        initial_datetime=initial_datetime,
        distance_function=distance_function,
        distance_threshold=distance_threshold,
    )

    for frame in video_handler:
        # get detections from model inference
        # detections = model.predict(
        #     frame,
        #     imgsz=IMG_SZ,
        #     device=device,
        #     conf=CONF_THRESHOLD,
        #     iou=IOU_THRESHOLD,
        #     verbose=False,
        # )

        # # convert to format norfair can use NOTE: eventually this gets done inside
        # # of the model inference function
        # norfair_detections = yolo_detections_to_norfair_detections(detections)
        norfair_detections = model.inference(frame, device=device)
        valid_detections = []
        for detection in norfair_detections:
            top_left, top_right = detection.points[0], detection.points[1]
            if check_overlap(np.array(roi), np.array(bbox_to_polygon(*top_left, *top_right))):
                valid_detections.append(detection)

        # update tracker with new detections
        tracked_objects = vehicle_tracker.update(frame, valid_detections)

        # drawing stuff
        draw_polygon(frame, np.array(roi))
        draw_detector_predictions(frame, norfair_detections, track_points="bbox")
        draw_tracker_predictions(frame, tracked_objects, track_points="bbox")
        draw_tracklets(frame, path_drawer, tracked_objects)

        if save_video:
            video_handler.write_frame_to_video(frame)

        if debug and video_handler.show(frame, 10) == ord("q"):
            video_handler.cleanup()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track vehicles in video and obtain data about them."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="model weight filename in form <filename>.pt",
        default="models/detection/yolov8/small_training.pt",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="video filename in form <filename>.<ext>",
        default="test_videos/short_video.mp4",
    )
    parser.add_argument(
        "--debug",
        help="show frames and print statements while processing",
        action="store_true",
    )
    parser.add_argument("--save", help="write video to a file", action="store_true")
    args = parser.parse_args()
    model_path = args.model
    video_path = args.video
    debug = args.debug
    save = args.save

    # # TODO: use model factories here for models
    # model = YOLO(model_path, task="detect")

    model_selector = ModelSelector(ROOT_DIR / "models/")
    valid_list = model_selector.search("detect")
    print(valid_list)
    metadata = valid_list[0]
    model = model_selector.generate_model(metadata)

    # setting up the video for reading/writing frames
    video_output_dir = str(ROOT_DIR / "output_videos")
    video_handler = VideoHandler(input_path=video_path, output_dir=video_output_dir)

    import time  # NOTE: FOR TESTING ...................................

    start = time.time()
    process(model, video_handler, save, debug)
    print(time.time() - start)
