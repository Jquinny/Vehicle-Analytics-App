from pathlib import Path

# TODO FIGURE OUT HOW TO DEAL WITH MULTIPLE DIFFERENT MODEL CLASSES

ROOT_DIR = Path(__file__).parent

# detector labels and their human readable names
DETECTOR_NUM2NAME = {0: "truck"}
DETECTOR_NAME2NUM = {val: key for key, val in DETECTOR_NUM2NAME.items()}

# classifier labels and their human readable names
CLASSIFIER_NUM2NAME = {
    0: "LT",
    1: "LT-C",
    2: "TrOnly",
    3: "TrChass",
    4: "TrFlat",
    5: "TrFlat",
    6: "TrTrail",
    7: "TrMarine",
    8: "TrRail",
    9: "TrTank",
    10: "Const",
    11: "Waste",
    12: "O",
    13: "TrReefer",
}
CLASSIFIER_NAME2NUM = {val: key for key, val in CLASSIFIER_NUM2NAME.items()}

# TODO: UPDATE THESE
VEHICLE_DATA_COLUMNS = {
    "timestamp": "string",
    "video_timestamp": "string",
    "initial_frame_index": "UInt64",
    "num_of_frames": "UInt64",
    "class": "string",
    "confidence": "Float64",
    "entry_direction": "string",
    "exit_direction": "string",
    "speed": "Float64",
    "color": "string",
}
