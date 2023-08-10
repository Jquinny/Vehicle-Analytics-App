# Models

This directory contains all trained models as well as a metadata file associated
with each one. Directory structure is split according to task and then according
to model architecture.

### JSON Format

```
{
    "model_filename": <string>,
    "architecture": <string>,
    "task": <string>,
    "classes": {
        <string>: <int>,
    },
    "precision": <float>,
    "recall": <float>,
    "mAP": <float>,
    "num_of_params": <int>,
    "hyperparams": {
        <string>: <value>,
    },
    "setup_params": {
        <string>: <value>,
    },
    "inference_params": {
        <string>: <value>,
    }
}
```

##### Example

```
{
    "model_filename": "testing.pt",
    "architecture": "yolov8",
    "task": "detect",
    "classes": {
        "LT": 0,
        "TrOnly": 1,
        "TrChass": 2,
        "TrFlat": 3
    },
    "precision": 0.9,
    "recall": 0.83,
    "mAP": 0.93,
    "num_of_params": 12345,
    "hyperparams": {
        "imgsz": 512,
        "learning_rate": 0.01,
        "optimizer": "Adam"
    },
    "setup_params": null,
    "inference_params": {
        "imgsz": 512,
        "conf": 0.45,
        "iou": 0.7
    }
}
```
