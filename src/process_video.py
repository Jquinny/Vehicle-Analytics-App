"""Main script for running the full vehicle analysis pipeline."""
from typing import List, Tuple, Dict, Any, Optional
from contextlib import contextmanager
from pathlib import Path, PosixPath, WindowsPath
import pathlib
import argparse
import time
import warnings

import pandas as pd
import numpy as np
import torch

import cv2 as cv
import easyocr
from norfair.tracker import Tracker, TrackedObject, Detection
from norfair.utils import print_objects_as_table  # for testing ............
from norfair.drawing import Palette

from config import ROOT_DIR, VEHICLE_DATA_COLUMNS
from src.tracking import VehicleInstance, VehicleDetection
from src.models.base_model import BaseModel
from src.utils.user_input import get_roi, draw_roi, get_coordinates
from src.utils.drawing import (
    draw_roi,
    draw_coordinates,
    draw_detector_predictions,
    draw_rect,
    draw_tracker_predictions,
    draw_zones,
    plot_images,
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
PAST_DETECTIONS_LENGTH: int = 5

# detection constants
CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50

ROI_OVERLAP_PCT: float = 0.15


# required for unpickling pytorch models sometimes if they were trained on
# a linux/mac machine
@contextmanager
def set_posix_windows():
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


def ENMS(
    classes: List[int],
    confidences: List[float],
    feature_vecs: List[Any],
    similarity_thresh: float = 0.5,
) -> float:
    """computes the image ..."""
    pass


def process(
    video_path: str,
    detector: BaseModel,
    classifier: BaseModel | None = None,
    active_learning: bool = False,
    active_learning_classes: List[int] | None = None,
    save_video: bool = False,
    debug: bool = False,
) -> str:
    """Main video processing function. accumulates vehicle data and writes out
    the video with bounding boxes, class names, and confidences, as well as
    a .json file with video metadata and a .csv file with all of the vehicle
    information.

    Arguments
    ---------
    video_path (str):
        absolute path to the video to be processed
    detector (BaseModel):
        the object detector used for vehicle detection and classification
    classifier (BaseModel | None):
        allows for running detector + classifier for better results
    active_learning (bool):
        whether to save low confidence frames for active learning
    active_learning_classes: (List[int] | None):
        the classes to be captured when doing active learning
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
    video_path = Path(video_path)
    output_dir = ROOT_DIR / "output/" / video_path.stem
    try:
        output_dir.mkdir(parents=True)
    except:
        warnings.warn("output directory already exists, overwriting previous data")
    video_handler = VideoHandler(input_path=str(video_path), output_dir=str(output_dir))

    # setup ocr model
    reader = easyocr.Reader(["en"])

    # setting the distance parameters for the tracker
    distance_function = "iou"
    distance_threshold = DISTANCE_THRESHOLD_BBOX

    first_frame = video_handler.get_frame(0)

    # try and grab timestamp from the video roi, direction coordinates, and timestamp
    initial_datetime = parse_timestamp(first_frame, reader)

    # get roi from the user
    roi = get_roi(first_frame)
    # if roi:
    #     roi_mask = np.zeros(frame.shape[:2], dtype="uint8")
    #     cv.fillPoly(
    #         roi_mask,
    #         [np.array([pt.as_tuple() for pt in roi.to_int()])],
    #         (255, 255, 255),
    #     )

    # get direction coordinates from the user
    draw_roi(first_frame, roi, close=True)
    direction_coordinates = get_coordinates(first_frame)

    # setup entry and exit zones based on user specified direction coordinates
    zone_mask: np.ndarray = None
    zone_mask_map: Dict[int, str] = None
    if direction_coordinates:
        zone_mask_map = {
            idx: dir for idx, dir in enumerate(direction_coordinates.keys())
        }

        # calculate entry/exit zone mask
        pixel_coords = np.indices(first_frame.shape[:2])[::-1].transpose(1, 2, 0)
        zone_coords = np.array([pt.as_tuple() for pt in direction_coordinates.values()])
        distances = np.linalg.norm(
            pixel_coords[:, :, None, :] - zone_coords[None, None, :, :], axis=-1
        )
        zone_mask = np.argmin(distances, axis=-1)

    tracker = Tracker(
        distance_function=distance_function,
        distance_threshold=distance_threshold,
        hit_counter_max=HIT_COUNTER_MAX,
        initialization_delay=INITIALIZATION_DELAY,
        past_detections_length=PAST_DETECTIONS_LENGTH,
    )

    # for storing vehicle instances
    vehicles: Dict[int, VehicleInstance] = dict()

    # for storing csv row results
    results: Dict[int, Dict[str, Any]] = dict()

    # for storing active learning frame numbers and the corresponding image frame
    active_learning_frames: Dict[int, np.ndarray] = {}

    # color map for visualizing bounding boxes and class estimates
    # NOTE: supports 20 classes for now
    Palette.set("tab20")

    print("beginning to process ...")
    start = time.time()
    # try:
    for frame_idx, frame in enumerate(video_handler):
        # # testing masking frame
        # if roi:
        #     frame = cv.bitwise_and(frame, frame, mask=roi_mask)

        # get detections from model inference
        norfair_detections = detector.inference(
            frame,
            device=device,
            verbose=False,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
        )

        # filter detections outside of ROI
        if roi is not None:
            valid_detections: List[Detection] = []
            for detection in norfair_detections:
                bbox = points_to_rect(detection.points)
                if roi.check_overlap(bbox.as_polygon(), overlap_pct=ROI_OVERLAP_PCT):
                    valid_detections.append(detection)
        else:
            # no roi means all are valid
            valid_detections: List[Detection] = norfair_detections

        for det in valid_detections:
            # add frame index to detection objects data for future processing
            det.data["frame_idx"] = frame_idx

        # update tracker with new valid detections
        tracked_objects: List[TrackedObject] = tracker.update(valid_detections)

        # update vehicle state
        for obj in tracked_objects:
            if obj.global_id not in vehicles.keys():
                # initialize new vehicle instance
                elapsed_time = int(frame_idx / video_handler.fps)
                vehicles[obj.global_id] = VehicleInstance(
                    initial_dt=initial_datetime,
                    elapsed_time=elapsed_time,
                    initial_frame_index=video_handler.current_frame,
                    detector_class_map=detector.get_classes(),
                )
            vehicles[obj.global_id].increment_frame_count()

            # update detection frames if active learning is on or using two-stage
            if classifier is not None or active_learn:
                vehicles[obj.global_id].update_detections(obj.past_detections)

            # hacky way to check if this obj actually got a match on this frame
            # or if they are just still alive but with no new information to give
            if obj.last_detection.age == obj.age:
                class_num = obj.last_detection.data.get("class", None)
                conf = obj.last_detection.data.get("conf", None)
                vehicles[obj.global_id].update_class_estimate(class_num, conf)

                # can use last detection to get better estimate of centroid
                center = points_to_rect(obj.last_detection.points).center
                vehicles[obj.global_id].update_coords(center)
            else:
                # approximate centroid of vehicle using kalman filter state
                center = points_to_rect(obj.estimate).center
                vehicles[obj.global_id].update_coords(center)

        # purge any vehicles no longer being tracked and store data in results
        for global_id, vehicle in list(vehicles.items()):
            if global_id not in [obj.global_id for obj in tracked_objects]:
                vehicle.compute_directions(
                    zone_mask,
                    zone_mask_map,
                    video_handler.resolution[0] - 1,
                    video_handler.resolution[1] - 1,
                )
                vehicle.classify(classifier)
                if active_learning:
                    # TODO: store first, middle, and last frame of past detections
                    # in a folder in the output directory. Make sure to
                    # store them in the yolo .txt format
                    pass
                if classifier is not None:  # for testing
                    plot_images([det.img for det in vehicle._detections])  # for testing
                results[global_id] = vehicle.get_data()
                del vehicles[global_id]

        # -------------- drawing output and whatnot below ---------------------
        frame_copy = frame.copy()
        if debug or save_video:
            # draw necessary info
            for detection in valid_detections:
                bbox = points_to_rect(detection.points)
                class_num = detection.data.get("class")
                conf = detection.data.get("conf")
                draw_rect(
                    frame_copy,
                    rect=bbox,
                    class_name=detector.get_classes().get(class_num),
                    conf=conf,
                    color=Palette.choose_color(class_num),
                )
            draw_roi(frame_copy, roi, close=True)
            draw_coordinates(frame_copy, direction_coordinates)
            video_handler.write_frame_to_video(frame_copy)

        if debug:
            # drawing other information that may be useful
            draw_tracker_predictions(frame_copy, tracked_objects)
            if zone_mask is not None:
                draw_zones(frame_copy, zone_mask)

        if debug and video_handler.show(frame_copy, 10) == ord("q"):
            video_handler.cleanup()
            break

    results_df = pd.DataFrame(
        [data for _, data in results.items()],
        columns=VEHICLE_DATA_COLUMNS,
    ).astype(dtype=VEHICLE_DATA_COLUMNS)
    # except Exception as e:
    #     print(e)
    #     # want to save anything processed so far in case program errors
    #     # out for whatever reason
    #     # results = vehicle_tracker.get_results()
    print(f"Total Time: {time.time() - start}")  # FOR TESTING ....................

    # write results to csv
    csv_path = str(output_dir / (video_path.stem + ".csv"))
    results_df.to_csv(csv_path, header=True, index=False, mode="w")

    return str(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track vehicles in video and obtain data about them."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="relative path to directory containing model metadata and weights",
        default="models/detection/test_yolov8n",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="relative path to video input file",
        default="videos/test/vehicle-counting.mp4",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="list of classes to pull frames for active learning",
    )
    parser.add_argument(
        "--active-learn",
        help="save low confidence frames for active learning",
        action="store_true",
    )
    parser.add_argument(
        "--two-stage",
        help="whether to run detector + classifier or just detector",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        help="show frames and print statements while processing",
        action="store_true",
    )
    parser.add_argument("--save", help="write video to a file", action="store_true")
    args = parser.parse_args()
    model_dir = Path(ROOT_DIR / args.model)
    video_path = str(ROOT_DIR / args.video)
    active_learn = args.active_learn
    debug = args.debug
    save = args.save

    detector_selector = ModelRegistry(str(model_dir.parent))
    detector = detector_selector.generate_model(model_dir.stem)

    classifier = None
    if args.two_stage:
        DEFAULT_CLASSIFIER = "2023-08-24_01-16-58"
        with set_posix_windows():
            classifier_selector = ModelRegistry(str(ROOT_DIR / "models/classification"))
            classifier = classifier_selector.generate_model(DEFAULT_CLASSIFIER)

    output_dir = process(
        video_path=video_path,
        detector=detector,
        classifier=classifier,
        active_learning=args.active_learn,
        active_learning_classes=args.classes,
        save_video=args.save,
        debug=args.debug,
    )
    print(f"\nOutput Directory: {output_dir}")
