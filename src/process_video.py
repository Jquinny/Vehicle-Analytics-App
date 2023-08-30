"""Main script for running the full vehicle analysis pipeline."""
from typing import List, Dict, Any
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
import pathlib
import json
import argparse
import warnings

import pandas as pd
import numpy as np
import torch

import easyocr
from norfair.tracker import Tracker, TrackedObject, Detection
from norfair.drawing import Palette

from config import ROOT_DIR, VEHICLE_DATA_COLUMNS
from src.active_learning import active_learn, write_yolo_annotation_files
from src.tracking import VehicleInstance
from src.models.base_model import BaseModel
from src.utils.user_input import get_roi, get_coordinates, check_existing_user_input
from src.utils.drawing import (
    draw_roi,
    draw_coordinates,
    draw_rect,
    draw_tracker_predictions,
    draw_zones,
)
from src.utils.geometry import points_to_rect, Point, Poly
from src.utils.image import parse_timestamp, extract_objects
from src.utils.video import VideoHandler
from src.model_registry import ModelRegistry

# tracking constants
INITIALIZATION_DELAY: int = 5
HIT_COUNTER_MAX: int = 15
DISTANCE_FUNCTION: str = "iou"
DISTANCE_THRESHOLD_BBOX: float = 0.5
PAST_DETECTIONS_LENGTH: int = 1

# detection constants
CONF_THRESHOLD = 0.30

ROI_OVERLAP_PCT: float = 0.15

DETECTOR_ROOT_DIR = str(ROOT_DIR / "models/detection")
CLASSIFIER_ROOT_DIR = str(ROOT_DIR / "models/classification")
DEFAULT_CLASSIFIER = "2023-08-24_01-16-58"


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


def process(
    video_path: str,
    detector_dir: str,
    two_stage: bool = False,
    active_learning: bool = False,
    active_learning_classes: List[int] | None = None,
    active_learning_budget: int = 100,
    save_video: bool = False,
    debug: bool = False,
) -> str:
    """Main video processing function. accumulates vehicle data and writes out
    the video with bounding boxes, class names, and confidences, as well as
    a .csv file with all of the vehicle information.

    If active learning is toggled, the algorithm will select optimal frames
    for training and compile them into the yolov8.txt dataset form so that
    they can easily be uploaded to any annotation software.

    Arguments
    ---------
    video_path (str):
        absolute path to the video to be processed
    detector_dir (str):
        the directory name holding the model weights and metadata to be used for
        detection
    two_stage (bool):
        whether to run the algorithm using detector + classifier or just detector
    active_learning (bool):
        whether to save low confidence frames for active learning
    active_learning_classes: (List[str] | None):
        the classes to be captured when doing active learning
    active_learning_budget (int):
        the max amount of images that can be extracted for active learning
        from this video
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
    # ------------------ Initial Setup -------------------------------------- #
    if debug:
        print("Setting up ...")

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

    reader = easyocr.Reader(["en"])

    # ------------------- User Input Setup ---------------------------------- #
    # try and grab timestamp from the first video frame
    first_frame = video_handler.get_frame(0)
    initial_datetime = parse_timestamp(first_frame, reader)

    # check if this camera view already has roi and coordinates
    json_file_list = list(video_path.parent.glob("user_input.json"))
    use_existing = False
    if json_file_list:
        use_existing = check_existing_user_input()

    if use_existing:
        # load in existing roi and direction coordinates
        with open(str(json_file_list[0]), "r") as f:
            data = json.load(f)
            roi = Poly([Point(x=x, y=y) for x, y in data.get("roi")])
            direction_coordinates = {
                dir: Point(*pt) for dir, pt in data.get("coordinates").items()
            }
    else:
        # get roi from the user
        roi = get_roi(first_frame)

        # get direction coordinates from the user
        draw_roi(first_frame, roi, close=True)
        direction_coordinates = get_coordinates(first_frame)

        # save new json holding the roi and coordinate information
        json_file = str(video_path.parent / "user_input.json")
        with open(json_file, "w") as f:
            data = {
                "roi": [pt.as_tuple() for pt in roi] if roi is not None else [],
                "coordinates": {
                    dir: pt.as_tuple() for dir, pt in direction_coordinates.items()
                },
            }
            json.dump(data, f)

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

    # ------------ detection and optionally classifier model setup ----------- #
    detector_selector = ModelRegistry(DETECTOR_ROOT_DIR)
    detector = detector_selector.generate_model(detector_dir)

    classifier = None
    if two_stage:
        with set_posix_windows():
            classifier_selector = ModelRegistry(CLASSIFIER_ROOT_DIR)
            classifier = classifier_selector.generate_model(DEFAULT_CLASSIFIER)

    cls_map = detector.get_classes() if classifier is None else classifier.get_classes()

    # -------------------- active learning setup -------------------------- #
    if active_learn and active_learning_classes:
        # map the active learning classes to their numerical form
        reverse_cls_map = {name: num for num, name in cls_map.items()}
        active_learning_classes = [
            reverse_cls_map[name]
            for name in active_learning_classes
            if name in reverse_cls_map.keys()
        ]

    # for storing active learning frames and their corresponding detections
    active_learning_frames: defaultdict[int, Dict[str, Any]] = defaultdict(lambda: {})

    # sampling rates for grabbing candidate images
    sampling_rate = video_handler.fps
    sample_tracker = 0

    # ----------------- tracking setup ------------------------------------- #
    distance_function = "iou"
    distance_threshold = DISTANCE_THRESHOLD_BBOX

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

    # color map for visualizing bounding boxes and class estimates
    # NOTE: supports 20 classes for now
    Palette.set("tab20")

    print("beginning to process ...")
    try:
        for frame_idx, frame in enumerate(video_handler):
            sample_tracker += 1

            # get detections from model inference
            norfair_detections = detector.inference(
                frame,
                device=device,
                verbose=False,
                conf=CONF_THRESHOLD,
            )

            # filter detections outside of ROI
            if roi is not None:
                valid_detections: List[Detection] = []
                for detection in norfair_detections:
                    bbox = points_to_rect(detection.points)
                    if roi.check_overlap(
                        bbox.as_polygon(), overlap_pct=ROI_OVERLAP_PCT
                    ):
                        valid_detections.append(detection)
            else:
                # no roi means all are valid
                valid_detections: List[Detection] = norfair_detections

            if valid_detections and two_stage:
                # run classifier on each single object and update class estimates
                single_obj_images = extract_objects(valid_detections, frame)
                for idx, img in enumerate(single_obj_images):
                    class_estimates = classifier.inference(img, verbose=False)
                    cls_num = np.argmax(class_estimates)
                    valid_detections[idx].data["class"] = cls_num
                    valid_detections[idx].data["conf"] = class_estimates[cls_num]

            # store frame for active learning if conditions are met
            if valid_detections and active_learning and sample_tracker > sampling_rate:
                active_learning_frames[frame_idx]["detections"] = valid_detections
                sample_tracker = 0

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
                        cls_map=cls_map,
                    )
                vehicles[obj.global_id].increment_frame_count()

                # hacky way to check if this obj actually got a match on this frame
                # or if they are just still alive but with no new information to give
                if obj.last_detection.age == obj.age:
                    # update the class estimate bin averages
                    class_num = obj.last_detection.data.get("class", None)
                    conf = obj.last_detection.data.get("conf", 0)
                    vehicles[obj.global_id].update_class_estimate(class_num, conf)

                    # estimate vehicle centroid using center of bounding box
                    center = points_to_rect(obj.last_detection.points).center
                    vehicles[obj.global_id].update_coords(center)

            # purge any vehicles no longer being tracked and store data in results
            for global_id, vehicle in list(vehicles.items()):
                if global_id not in [obj.global_id for obj in tracked_objects]:
                    # compute necessary vehicle metadata for extraction
                    vehicle.compute_directions(
                        zone_mask,
                        zone_mask_map,
                        video_handler.resolution[0] - 1,
                        video_handler.resolution[1] - 1,
                    )
                    vehicle.classify()

                    results[global_id] = vehicle.get_data()
                    del vehicles[global_id]

            # -------------- drawing output and whatnot below -------------------- #
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
                        class_name=cls_map.get(class_num),
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
                break

        results_df = pd.DataFrame(
            [data for _, data in results.items()],
            columns=VEHICLE_DATA_COLUMNS,
        ).astype(dtype=VEHICLE_DATA_COLUMNS)
    except Exception as e:
        print(e)
        # want to save anything processed so far in case program errors
        # out for whatever reason
        results_df = pd.DataFrame(
            [data for _, data in results.items()],
            columns=VEHICLE_DATA_COLUMNS,
        ).astype(dtype=VEHICLE_DATA_COLUMNS)

    # write results to csv
    csv_path = str(output_dir / (video_path.stem + ".csv"))
    results_df.to_csv(csv_path, header=True, index=False, mode="w")

    # only continue to diverse prototype algorithm if we actually have
    # frames to run it on
    if active_learning and active_learning_frames:
        print("Acquiring images for active learning ...")
        image_info_dict = active_learn(
            image_info=active_learning_frames,
            video_handler=video_handler,
            all_classes=list(cls_map.keys()),
            minority_classes=active_learning_classes,
            budget=active_learning_budget,
        )

        img_dir = output_dir / "saved_annotations/images"
        if not img_dir.exists():
            img_dir.mkdir(parents=True)

        for frame_idx, image_info in image_info_dict.items():
            frame = video_handler.get_frame(frame_idx)
            detections = image_info.get("detections")

            filename = video_path.stem + f"_{frame_idx}.jpeg"
            abs_img_path = img_dir / filename
            write_yolo_annotation_files(detections, cls_map, frame, str(abs_img_path))

    video_handler.cleanup()

    return str(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track vehicles in video and obtain data about them."
    )
    parser.add_argument(
        "video",
        type=str,
        help="absolute path to the video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="name of directory containing detector metadata and weights",
        default="test_yolov8n",
    )
    parser.add_argument(
        "--active-learn",
        help="save low confidence frames for active learning",
        action="store_true",
    )
    parser.add_argument(
        "--active-learning-classes",
        type=str,
        nargs="+",
        help="list of classes to pull frames for active learning",
    )
    parser.add_argument(
        "--active-learning-budget",
        type=int,
        help="the max amount of images that can be extracted for active learning from this video",
        default=100,
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
    video_path = str(Path(args.video).resolve())

    output_dir = process(
        video_path=video_path,
        detector_dir=args.model,
        two_stage=args.two_stage,
        active_learning=args.active_learn,
        active_learning_classes=args.active_learning_classes,
        active_learning_budget=args.active_learning_budget,
        save_video=args.save,
        debug=args.debug,
    )
    print(f"\nOutput Directory: {output_dir}")
