from pathlib import Path
from typing import List

import json
import pandas as pd

from src.models.base_model import BaseModel
from src.models.detector import (
    YOLOv8Detector,
    RTDETRDetector,
    FasterRCNNDetector,
    EfficientDetDetector,
)
from src.models.classifier import (
    TestingClassifier,
    YOLOv8Classifier,
    ResNetClassifier,
    ViTClassifier,
    ClipClassifier,
)


class ModelRegistry:
    """interface into the models/ directory to select models and view model metadata

    The workflow is as follows:
        1.  initialize the model registry by passing it the absolute path to the
            base directory it should look in for finding model folders
        2.  create a pandas dataframe using the create_dataframe() function. this
            will create a dataframe with the model folder name and some metrics
            in each row
        3.  generate a model based on its model folder name using the
            generate_model() function, passing it the model folder name

    This will generate a model deriving from the BaseModel class, so that you
    can be sure it will abide by the BaseClass interface
    """

    def __init__(self, base_dir: str):
        """initializes model registry with

        Arguments
        ---------
        base_dir (str):
            the absolute path to the directory containing all of the models
            and their metadata files
        """
        self.base_dir = Path(base_dir)
        assert self.base_dir.exists()

        self._models = {}
        for dir in self.base_dir.iterdir():
            abs_dir = dir.resolve()
            metadata_path = abs_dir / "metadata.json"
            weight_path = abs_dir / "weights/best.pt"

            # check that both necessary files exist
            if not metadata_path.exists() or not weight_path.exists():
                continue

            with open(str(metadata_path.resolve()), "r") as f:
                metadata = json.load(f)
                self._models[str(abs_dir)] = {}
                self._models[str(abs_dir)]["weights"] = str(weight_path.resolve())
                self._models[str(abs_dir)]["metadata"] = metadata

    def create_dataframe(self) -> pd.DataFrame:
        """creates a pandas dataframe containing model info

        NOTE: this is meant to display the detectors, no support for classifiers
        """

        columns = ["folder", "mAP50", "inf speed (ms)", "num of classes"]

        model_info_list = []
        for abs_folder_dir, model_info in self._models.items():
            model_row = {}

            model_row["folder"] = Path(abs_folder_dir).stem

            metrics = model_info["metadata"].get("metrics", None)
            classes = model_info["metadata"].get("classes", None)
            map = metrics.get("mAP50", None) if metrics else None
            inf_speed = (
                metrics.get("avg_inference_time (ms)", None) if metrics else None
            )
            model_row["mAP50"] = map
            model_row["inf speed (ms)"] = inf_speed

            num_of_classes = len(classes) if classes else None
            model_row["num of classes"] = num_of_classes

            model_info_list.append(model_row)

        return pd.DataFrame(model_info_list, columns=columns)

    def generate_model(self, model_folder: str) -> BaseModel:
        """generates a model based on the info stored in the registry.

        Arguments
        ---------
        model_dir (str):
            the name of the directory, inside the base model directory, containing
            the model to be created and its metadata

        Returns
        -------
        BaseModel:
            a model object implementing the BaseModel interface

        Raises
        ------
        ValueError:
            if the model folder is invalid
        """
        model_folder_path = (self.base_dir / model_folder).resolve()
        model_info = self._models.get(str(model_folder_path), None)
        if model_info:
            model_arch = model_info["metadata"].get("architecture", None)
            model_task = model_info["metadata"].get("task", None)
            model = ModelFactory.create(model_arch=model_arch, task=model_task)
            model.setup(model_path=model_info["weights"], params=model_info["metadata"])
        else:
            raise ValueError("Invalid model folder")

        return model


class ModelFactory:
    """generic model factory for generating different types of models like
    detectors, classifiers, and possibly even ReID models
    """

    @staticmethod
    def create(model_arch: str, task: str) -> BaseModel:
        """factory method for generating model objects based on a specified
        model architecture

        Arguments
        ---------
        model_arch (str):
            the model architecture
        task (str):
            the task being performed by the model
        """

        if task == "detect":
            if "yolov8" in model_arch:
                return YOLOv8Detector()
            elif "rtdetr" in model_arch:
                return RTDETRDetector()
            else:
                raise ValueError("invalid detector architecture")
        elif task == "classify":
            if "yolov8" in model_arch:
                return YOLOv8Classifier()
            else:
                raise ValueError("invalid classifier architecture")
        else:
            raise ValueError("invalid model task")
