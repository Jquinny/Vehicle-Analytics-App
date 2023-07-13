"""Main script for running the full truck analysis pipeline.

TODO:
-   clean this up and fix errors (i.e. actually figure out how to interpret
    yolo results and work with norfair tracking)
"""

import argparse
import os
from typing import List, Optional, Union

import numpy as np
import torch

import cv2 as cv
import norfair
from norfair import Detection, Paths, Tracker, Video
from ultralytics import YOLO

DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000
IMG_SZ = 720
CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.45

LABEL2NAME = {0: "lower_class", 1: "upper_class"}


def center(points):
    return [np.mean(np.array(points), axis=0)]


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    # TODO: figure this out
    # if track_points == "centroid":
    # for detection in yolo_detections:
    #     bbox_as_xywh = detection.boxes.cpu().numpy().xywh.astype(int)
    #     score = detection.boxes.cpu().numpy().conf
    #     cls = detection.boxes.cpu().numpy().cls.astype(int)
    #     if len(bbox_as_xywh) > 0:
    #         centroid = np.array(
    #             [
    #                 bbox_as_xywh[0, 0] + bbox_as_xywh[0, 2] // 2,
    #                 bbox_as_xywh[0, 1] + bbox_as_xywh[0, 3] // 2,
    #             ]
    #         )
    #         norfair_detections.append(
    #             Detection(
    #                 points=centroid,
    #                 scores=score,
    #                 label=LABEL2NAME[cls[0]],
    #             )
    #         )
    # elif track_points == "bbox":
    for detection in yolo_detections[0].boxes:
        bbox_as_xyxy = detection.cpu().numpy().xyxy.astype(int)
        score = detection.cpu().numpy().conf
        cls = detection.cpu().numpy().cls.astype(int)
        bbox = np.array(
            [
                [bbox_as_xyxy[0, 0], bbox_as_xyxy[0, 1]],
                [bbox_as_xyxy[0, 2], bbox_as_xyxy[0, 3]],
            ]
        )
        score_per_corner = np.ones(2, dtype=int) * score
        norfair_detections.append(
            Detection(
                points=bbox,
                scores=score_per_corner,
                label=LABEL2NAME[cls[0]],
            )
        )

    return norfair_detections


def merge_frames(track_mask, video_frame):
    return cv.addWeighted(track_mask, 1, video_frame, 1, 0)


def main(model_path, video_path, track_points="bbox"):
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

    for frame in video:
        # get detections from model inference
        yolo_detections = model.predict(
            frame,
            imgsz=IMG_SZ,
            device=device,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
        )

        # convert to format norfair can use
        detections = yolo_detections_to_norfair_detections(
            yolo_detections, track_points=track_points
        )

        # update tracker with new detections
        tracked_objects = tracker.update(detections=detections)

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
        default="demo_weights.pt",
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

    main(model_path, video_path, track_points)
