# Models

This directory contains all trained models as well as a metadata file associated
with each one. Each directory in the task specific folders should contain
the metadata.json file and the model weights in weights/best.pt

The directory structure should look as follows:

```
models
    |
    ├── <task 1 i.e. detection, classification, re-identification, etc.>
    |       |
    |       ├── <model 1>
    |       |       ├── <weights>
    |       |       |       └── <best.pt>
    |       |       └── <metadata.json>
    |       .
    |       .
    |       ├── <model n>
    |       |       ├── <weights>
    |       |       |       └── <best.pt>
    |       |       └── <metadata.json>
```

### JSON Format

```
{
    "architecture": <string>,
    "task": <string>,
    "hyper_params": {
        <string>: <value>,
    },
    "classes": {
        <int>: <string>,
    },
    "metrics": {
        <string>: <value>
    }
}
```

##### Example

```
{
    "architecture": "yolov8n.pt",
    "task": "detect",
    "classes": {
        "0": "bus",
        "1": "car",
        "2": "truck",
    },
    "hyper_params": {
        "model_type": "yolov8n.pt",
        "pretrained": true,
        "device": 0,
        "imgsz": 640,
        "batch": 16,
        "epochs": 100,
        "optimizer": "Adam",
        "lr0": 0.001,
        "data": "~/dataset/data.yaml"
    },
    "metrics": {
        "avg_inference_time (ms)": 15.62,
        "precision": 0.83,
        "recall": 0.80,
        "mAP50": 0.87,
        "mAP50-95": 0.74,
        "fitness": 0.75
    }
}
```
