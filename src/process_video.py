"""Main script for running the full vehicle analysis pipeline."""

from pathlib import Path
import argparse
import time
import warnings

import numpy as np
import torch

import cv2 as cv
import easyocr
from norfair.utils import print_objects_as_table  # for testing ............
from norfair.drawing import Palette

from config import ROOT_DIR
from src.tracking import VehicleInstanceTracker
from src.models.base_model import BaseModel
from src.utils.user_input import get_roi, draw_roi, get_coordinates
from src.utils.drawing import (
    draw_roi,
    draw_coordinates,
    draw_detector_predictions,
    draw_rect,
    draw_tracker_predictions,
    draw_zones,
)
from src.utils.geometry import points_to_rect
from src.utils.image import parse_timestamp
from src.utils.video import VideoHandler
from src.model_registry import ModelRegistry

# NOTE: play with these constants, or try an algorithm that somehow finds the
# best results once I have a ground truth set up

# tracking constants
INITIALIZATION_DELAY: int = 5
HIT_COUNTER_MAX: int = 15
DISTANCE_FUNCTION: str = "iou"
DISTANCE_THRESHOLD_BBOX: float = 0.7
MAX_DISTANCE: int = 10000  # not sure why I had this here
DETECTION_THRESHOLD: float = 0  # may be redundant if I already have CONF_THRESHOLD

# detection constants
# CONF_THRESHOLD = 0.45
# IOU_THRESHOLD = 0.45

ROI_OVERLAP_PCT: float = 0.15


def process(
    detector: BaseModel,
    video_path: str,
    save_video: bool = False,
    debug: bool = False,
) -> str:
    """Main video processing function. accumulates vehicle data and writes out
    the video with bounding boxes, class names, and confidences, as well as
    a .json file with video metadata and a .csv file with all of the vehicle
    information.

    Arguments
    ---------
    detector (BaseModel):
        the object detector used for vehicle detection and classification
    video_path (str):
        absolute path to the video to be processed
    save_video (bool):
        if true, save output video with bounding boxes and classes drawn on frames
    debug (bool):
        it true, displays each frame on the fly in an opencv window with all
        bounding boxes, classes, and tracker predictions drawn

    Returns
    -------
    str:
        path to the output directory
    """
    if debug:
        print("Setting up ...")

    # automatically set device for inferencing
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if debug:
        print(f"using device {device}")

    # setting up the output directory and the video handler
    video_filename = Path(video_path).stem
    output_dir = ROOT_DIR / "output/" / video_filename
    try:
        output_dir.mkdir(parents=True)
    except:
        warnings.warn("output directory already exists, overwriting previous data")
    video_handler = VideoHandler(input_path=video_path, output_dir=str(output_dir))

    # setup ocr model
    reader = easyocr.Reader(["en"])

    # setting the distance parameters for the tracker
    distance_function = "iou"
    distance_threshold = DISTANCE_THRESHOLD_BBOX

    # grab roi, direction coordinates, and timestamp
    frame = video_handler.get_frame(0)
    initial_datetime = parse_timestamp(frame, reader)
    roi = get_roi(frame)
    draw_roi(frame, roi, close=True)  # just so user sees ROI when selecting coords
    coord_points, coord_map = get_coordinates(frame)

    # calculate entry/exit zone mask
    pixel_coords = np.indices(frame.shape[:2])[::-1].transpose(1, 2, 0)
    zone_coords = np.array([pt.as_tuple() for pt in coord_points])
    distances = np.linalg.norm(
        pixel_coords[:, :, None, :] - zone_coords[None, None, :, :], axis=-1
    )
    zone_mask = np.argmin(distances, axis=-1)

    # set up vehicle tracker
    vehicle_tracker = VehicleInstanceTracker(
        video_handler=video_handler,
        class_map=detector.get_classes(),
        roi=roi,
        zone_mask=zone_mask,
        zone_mask_map=coord_map,
        initial_datetime=initial_datetime,
        distance_function=distance_function,
        distance_threshold=distance_threshold,
        hit_counter_max=HIT_COUNTER_MAX,
        initialization_delay=INITIALIZATION_DELAY,
    )

    # color map for visualizing bounding boxes and class estimates
    # NOTE: supports 20 classes for now
    Palette.set("tab20")

    if debug:
        print("beginning to process ...")

    start = time.time()
    try:
        for frame in video_handler:
            # get detections from model inference
            norfair_detections = detector.inference(frame, device=device, verbose=False)

            # filter detections outside of ROI
            valid_detections = []
            for detection in norfair_detections:
                bbox = points_to_rect(detection.points)
                if roi.check_overlap(bbox.as_polygon(), overlap_pct=ROI_OVERLAP_PCT):
                    valid_detections.append(detection)

            # update tracker with new detections
            tracked_objects = vehicle_tracker.update(frame, valid_detections)

            if debug or save_video:
                # draw necessary info
                for detection in valid_detections:
                    bbox = points_to_rect(detection.points)
                    class_num = detection.data.get("class")
                    conf = detection.data.get("conf")
                    draw_rect(
                        frame,
                        rect=bbox,
                        class_name=detector.get_classes().get(class_num),
                        conf=conf,
                        color=Palette.choose_color(class_num),
                    )
                draw_roi(frame, roi, close=True)
                draw_coordinates(frame, coord_points, coord_map)
                video_handler.write_frame_to_video(frame)

            if debug:
                # drawing other information that may be useful
                draw_tracker_predictions(frame, tracked_objects)
                draw_zones(frame, zone_mask)

            if debug and video_handler.show(frame, 10) == ord("q"):
                video_handler.cleanup()
                break

        results = vehicle_tracker.get_results()
    except Exception as e:
        print(e)
        # want to save anything processed so far in case program errors
        # out for whatever reason
        results = vehicle_tracker.get_results()
    print(f"Total Time: {time.time() - start}")  # FOR TESTING ....................

    # write results to csv
    csv_path = str(output_dir / (video_filename + ".csv"))
    results.to_csv(csv_path, header=True, index=False, mode="w")

    return str(output_dir)


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
        help="relative path to video input file",
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
    video_path = str(ROOT_DIR / args.video)
    debug = args.debug
    save = args.save

    detector_selector = ModelRegistry(ROOT_DIR / "models/detection")
    detector = detector_selector.generate_model("test")

    output_dir = process(detector, video_path, save, debug)
    print(f"\nOutput Directory: {output_dir}")
