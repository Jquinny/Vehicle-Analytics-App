"""module for training yolov8 on a user selected dataset"""

from pathlib import Path
from typing import Literal
import argparse
import datetime
import warnings
import random
import time
import shutil
import json

import torch
import yaml

from ultralytics import YOLO

from config import ROOT_DIR


def _verify_model(model_dir: str, model_arch: str) -> str | None:
    """helper for verifying a model directory contains the right size of model
    to use as the starting point

    Arguments
    ---------
    model_dir (str):
        absolute path to the model directory containing model metadata and model
        weights
    model_arch (str):
        the model architecture to look for in metadata.json or args.yaml

    Returns
    -------
    str:
        the absolute path to the model weights found in the model directory,
        or None if model not valid
    """

    if model_dir is None:
        return None

    model_dir = Path(model_dir)
    for item in model_dir.iterdir():
        if item.name == "metadata.json":
            # checking metadata file for model architecture
            with open(item, "r") as f:
                metadata = json.load(f)
                if metadata.get("architecture") == model_arch:
                    return str((model_dir / "weights/best.pt").resolve())
        if item.name == "args.yaml":
            # checking ultralytics generated file for model architecture
            with open(item, "r") as f:
                args = yaml.safe_load(f)
                if args.get("model") == model_arch:
                    return str((model_dir / "weights/best.pt").resolve())

    return None


def _grab_model(
    model_arch: Literal["yolov8s", "yolov8m", "yolov8l"],
    model_dir: str | None,
) -> YOLO:
    """creates a yolov8 small, medium, or large model from pretrained weights

    Arguments
    ---------
    model_arch (Literal["yolov8s", "yolov8m", "yolov8l"]):
        what size of yolov8 model to train
    model_dir (str | None):
        absolute path to the model directory containing the model metadata and
        the model weights to be used for a starting point when training

    Returns
    -------
    YOLO:
        the yolo model object to be trained

    Raises
    ------
    ValueError: if invalid model architecture is given
    """

    if model_arch == "yolov8s":
        # grab model weights or start from COCO pretrained if not there
        weights = _verify_model(model_dir, "yolov8s.pt")
        if weights is None:
            warnings.warn(
                f"{model_dir} does not contain a yolov8s model, training from COCO checkpoint"
            )
            weights = "yolov8s.pt"
        return YOLO(weights)
    elif model_arch == "yolov8m":
        # grab model weights or start from COCO pretrained if not there
        weights = _verify_model(model_dir, "yolov8m.pt")
        if weights is None:
            warnings.warn(
                f"{model_dir} does not contain a yolov8m model, training from COCO checkpoint"
            )
            weights = "yolov8m.pt"
        return YOLO(weights)
    elif model_arch == "yolov8l":
        # grab model weights or start from COCO pretrained if not there
        weights = _verify_model(model_dir, "yolov8l.pt")
        if weights is None:
            warnings.warn(
                f"{model_dir} does not contain a yolov8l model, training from COCO checkpoint"
            )
            weights = "yolov8l.pt"
        return YOLO(weights)
    else:
        raise ValueError(
            f"invalid model option {model_arch}\nOptions are fast, middle-ground, accurate"
        )


def train(
    model_dir: str | None,
    model_arch: Literal["yolov8s", "yolov8m", "yolov8l"],
    train_time: Literal["fast", "medium", "long"],
    dataset: str,
) -> str:
    """trains a yolov8 small, medium, or large model using pretrained weights
    as starting point

    Arguments
    ---------
    model_dir (str):
        absolute path to the model directory containing the model to use as a
        training starting point
    model_arch (Literal["yolov8s", "yolov8m", "yolov8l"]):
        the yolov8 model architecture to train
    train_time (Literal["fast", "medium", "long"]):
        how long to train the model for
    dataset (str):
        absolute path to the dataset directory that the user will be training on

    Returns
    -------
    str:
        path to the model folder generated in the models/ directory

    Raises
    ------
    ValueError: if invalid training time is given
    """

    # first grab the model object based on the directory specified
    model = _grab_model(model_arch=model_arch, model_dir=model_dir)

    if train_time == "fast":
        epochs = 1  # NOTE: put back to 50 when done testing
    elif train_time == "medium":
        epochs = 150
    elif train_time == "long":
        epochs = 300
    else:
        raise ValueError(
            f"invalid training time option: {train_time}\nOptions are fast, medium, and slow"
        )

    # now we need to find the yaml file
    yaml_files = list(Path(dataset).glob("*.yaml"))
    assert (
        len(yaml_files) == 1
    ), "invalid dataset configuration. Make sure to have one and only one yaml file"

    if torch.cuda.is_available():
        device = "cuda:0"
        torch.cuda.empty_cache()
    else:
        device = "cpu"

    hyper_params = {
        "pretrained": True,
        "device": device,
        "seed": 42,
        "patience": 15,
        "imgsz": 640,
        "batch": 2,
        "epochs": epochs,
        "optimizer": "Adam",
        "lr0": 0.001,
        "data": str(yaml_files[0]),
    }

    # metadata to be stored with the model training outputs
    model_metadata = {
        "architecture": model_arch + ".pt",
        "task": "detect",
        "hyper_params": hyper_params,
        "classes": model.names,
        "metrics": {},
    }

    # time to train the model
    name = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    project = "training_runs"
    model.train(**hyper_params, name=name, project=project)

    # grabbing test set metrics for model metadata
    metrics = model.val(
        split="test",
        device=device,
    )
    # getting metrics as python floats instead of numpy scalars
    proper_metrics = {
        key.replace("metrics/", "").replace("(B)", ""): float(val)
        for key, val in metrics.results_dict.items()
    }

    # sample 100 images from dataset and compute avg inference time
    test_image_dir = Path(hyper_params["data"]).parent / "test/images"
    img_file_list = list(test_image_dir.glob("*.jpg"))
    sample_num = 101 if len(img_file_list) > 101 else len(img_file_list)
    fps_testing_images = random.sample(img_file_list, sample_num)
    total_time = 0
    for img_num, img_file in enumerate(fps_testing_images):
        # first image takes longer for inference
        if img_num > 0:
            start_time = time.time()
            _ = model.predict(img_file, verbose=False, device=device)
            total_time += time.time() - start_time

    avg_inf_time = total_time / (len(fps_testing_images) - 1)

    # add inference info to model metadata
    model_metadata["metrics"] = proper_metrics
    model_metadata["metrics"]["avg_inference_time (ms)"] = avg_inf_time * 1000

    # moving ultralytics output files into appropriate directory
    ultralytics_save_dir = ROOT_DIR / f"{project}/{name}"
    models_dir = ROOT_DIR / f"models/detection/{name}"
    shutil.copytree(ultralytics_save_dir, models_dir)
    shutil.rmtree(ultralytics_save_dir.parent)

    # write out the metadata.json file
    metadata_path = str(models_dir / "metadata.json")
    with open(metadata_path, "w") as f:
        f.write(json.dumps(model_metadata, indent=4))

    return f"\npath to trained model: {str((models_dir).resolve())}\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track vehicles in video and obtain data about them."
    )
    parser.add_argument(
        "dataset", type=str, help="absolute path to the dataset to be used for training"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        help="relative path to the model directory containing weights and metadata",
        default=None,
    )
    parser.add_argument(
        "--train-time",
        type=str,
        help="how long to train for",
        choices=["fast", "medium", "long"],
        default="medium",
    )
    parser.add_argument(
        "--arch",
        type=str,
        help="yolov8 model architecture to train",
        choices=["yolov8s", "yolov8m", "yolov8l"],
        default="yolov8m",
    )
    args = parser.parse_args()
    model_dir = str(ROOT_DIR / args.model_dir) if args.model_dir else None
    train_time = args.train_time
    model_arch = args.arch
    dataset_path = args.dataset

    print(
        train(
            model_dir=model_dir,
            train_time=train_time,
            model_arch=model_arch,
            dataset=dataset_path,
        )
    )
