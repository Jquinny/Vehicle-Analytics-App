"""Main script for running the full vehicle analysis pipeline."""
from typing import List, Dict, Any
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
import pathlib
import json
import argparse
import time
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
from src.utils.image import parse_timestamp
from src.utils.video import VideoHandler
from src.model_registry import ModelRegistry

# tracking constants
INITIALIZATION_DELAY: int = 5
HIT_COUNTER_MAX: int = 15
DISTANCE_FUNCTION: str = "iou"
DISTANCE_THRESHOLD_BBOX: float = 0.5
PAST_DETECTIONS_LENGTH: int = 5

# detection constants
CONF_THRESHOLD = 0.60

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


def process(
    video_path: str,
    detector: BaseModel,
    classifier: BaseModel | None = None,
    active_learning: bool = False,
    active_learning_classes: List[int] | None = None,
    active_learning_budget: int = 100,
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

    distance_function = "iou"
    distance_threshold = DISTANCE_THRESHOLD_BBOX

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

    # for storing active learning frames and their corresponding detections
    active_learning_frames: defaultdict[int, Dict[str, Any]] = defaultdict(
        lambda: {"detections": []}
    )
    if active_learn:
        # make sure active learning classes are set up properly
        all_classes = (
            list(detector.get_classes().keys())
            if classifier is None
            else list(classifier.get_classes().keys())
        )
        if active_learning_classes is None:
            active_learning_classes = all_classes
        else:
            if not set(active_learning_classes).issubset(set(all_classes)):
                active_learning_classes = [
                    cls for cls in active_learning_classes if cls in all_classes
                ]

    # color map for visualizing bounding boxes and class estimates
    # NOTE: supports 20 classes for now
    Palette.set("tab20")

    cls_map = detector.get_classes() if classifier is None else classifier.get_classes()

    print("beginning to process ...")
    start = time.time()
    try:
        for frame_idx, frame in enumerate(video_handler):
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

            if active_learning:
                # need frame indexes stored with detections for active learning
                for det in valid_detections:
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

                # hacky way to check if this obj actually got a match on this frame
                # or if they are just still alive but with no new information to give
                if obj.last_detection.age == obj.age:
                    if classifier is None:
                        # only need to update detection bins if running single stage
                        class_num = obj.last_detection.data.get("class", None)
                        conf = obj.last_detection.data.get("conf", None)
                        vehicles[obj.global_id].update_class_estimate(class_num, conf)

                    # can use last detection to get better estimate of centroid
                    center = points_to_rect(obj.last_detection.points).center
                    vehicles[obj.global_id].update_coords(center)

                    # update detection frames if active learning is on or using two-stage
                    if classifier is not None or active_learning:
                        vehicles[obj.global_id].update_detections(obj.past_detections)
                else:
                    # approximate centroid of vehicle using kalman filter state
                    center = points_to_rect(obj.estimate).center
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
                    vehicle.classify(classifier)

                    # add frames for active learning computation if this vehicle
                    # class was toggled by the user
                    if (
                        active_learning
                        and vehicle.get_class_info()[0] in active_learning_classes
                    ):
                        for det in vehicle.get_detections():
                            frame_idx = det.data.get("frame_idx")
                            active_learning_frames[frame_idx]["detections"].append(det)

                    results[global_id] = vehicle.get_data(cls_map)
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
    print(f"Total Time: {time.time() - start}")  # FOR TESTING ....................

    # write results to csv
    csv_path = str(output_dir / (video_path.stem + ".csv"))
    results_df.to_csv(csv_path, header=True, index=False, mode="w")

    # only continue to diverse prototype algorithm if we actually have
    # frames to run it on
    if active_learning and active_learning_frames:
        image_info_dict = active_learn(
            image_info=active_learning_frames,
            all_classes=all_classes,
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
            write_yolo_annotation_files(detections, frame, str(abs_img_path))

    video_handler.cleanup()

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
        "--active-learn",
        help="save low confidence frames for active learning",
        action="store_true",
    )
    parser.add_argument(
        "--active-learning-classes",
        type=int,
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
    model_dir = Path(ROOT_DIR / args.model)
    video_path = str(ROOT_DIR / args.video)

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
        active_learning_classes=args.active_learning_classes,
        active_learning_budget=args.active_learning_budget,
        save_video=args.save,
        debug=args.debug,
    )
    print(f"\nOutput Directory: {output_dir}")
