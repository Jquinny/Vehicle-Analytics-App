from pathlib import Path
from typing import List

import json

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


class ModelSelector:
    """interface into the models/ directory to select models and view model metadata

    NOTE: for selecting models from local directories, this class assumes that
    all metadata files are in the same directory as the model they correspond
    to

    TODO: implement roboflow integration in case user's want to use a model
    being hosted on roboflow
    """

    def __init__(self, model_base: Path):
        """initializes model selector with all model metadata filepaths

        Arguments
        ---------
        model_base (Path):
            the absolute path to the base directory containing all of the models
            and their metadata files
        """

        def traverse_directory(dir: Path, filepaths: List[Path]):
            for item in dir.iterdir():
                if item.is_dir():
                    traverse_directory(item, filepaths)
                elif item.is_file() and item.suffix == ".json":
                    filepaths.append(item.resolve())

        self.metadata_paths = []
        traverse_directory(model_base, self.metadata_paths)

    def search(self, task: str) -> List[dict]:
        """check metadata files and extract ones for models that complete this task

        Arguments
        ---------
        task (str):
            the task to be accomplished by the model. Choices are:
            -   Detection
            -   Classification

        Returns
        -------
        List[dict]:
            list of dictionaries containing the metadata for each model that
            completes this task
        """

        valid_model_data = []
        for path in self.metadata_paths:
            with open(path, "r") as f:
                metadata = json.load(f)
                task_str = metadata.get("task")
                if task_str is None or task_str != task:
                    # model does not match task
                    continue

                model_name = metadata.get("model_filename")
                if model_name is None:
                    # missing model filename in the metadata file
                    continue

                model_path = path.parent / model_name
                if not model_path.exists():
                    # model isn't actually in the folder
                    continue

                # passed all checks, should be good to go
                metadata["model_path"] = str(model_path)
                valid_model_data.append(metadata)

        return valid_model_data

    def generate_model(self, metadata: dict) -> BaseModel:
        """generates a model object from the metadata dictionary provided"""
        assert (
            metadata.get("model_path") is not None
            and metadata.get("architecture") is not None
            and metadata.get("task") is not None
        ), "metadata is missing necessary information"

        model = ModelFactory.create(
            task=metadata["task"], model_arch=metadata["architecture"]
        )
        model.setup(model_path=metadata["model_path"], params=metadata)

        return model


class ModelFactory:
    """generic model factory for generating different types of models like
    detectors, classifiers, and possibly even ReID models
    """

    @staticmethod
    def create(task: str, model_arch: str) -> BaseModel:
        """factory method for generating model objects based on a specified
        task and model architecture

        Arguments
        ---------
        task (str):
            the task to be completed by the model. Choices include:
            -   detect
            -   classify
        model_arch (str):
            the model architecture. Choices include:
            Detection:
                -   yolov8
                -   rtdetr
                -   fasterrcnn
                -   efficientdet
            Classification:
                -   test
                -   yolov8
                -   resnet
                -   vit
                -   clip
        """
        if task == "classify":
            if model_arch == "test":
                return TestingClassifier()
            elif model_arch == "yolov8":
                return YOLOv8Classifier()
            elif model_arch == "resnet":
                return ResNetClassifier()
            elif model_arch == "vit":
                return ViTClassifier()
            elif model_arch == "clip":
                return ClipClassifier()
            else:
                raise ValueError("invalid model architecture")
        elif task == "detect":
            if model_arch == "yolov8":
                return YOLOv8Detector()
            elif model_arch == "rtdetr":
                return RTDETRDetector()
            elif model_arch == "fasterrcnn":
                return FasterRCNNDetector()
            elif model_arch == "efficientdet":
                return EfficientDetDetector()
            else:
                raise ValueError("invalid model architecture")
        else:
            raise ValueError("Invalid task type")
