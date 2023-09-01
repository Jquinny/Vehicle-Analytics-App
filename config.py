from pathlib import Path

ROOT_DIR = Path(__file__).parent

# This is here basically as a guideline for building the vehicle interfaces
# and for dataframe creation
VEHICLE_DATA_COLUMNS = {
    "timestamp": "string",
    "video_timestamp": "string",
    "initial_frame": "UInt64",
    "total_frames": "UInt64",
    "global_id": "UInt64",
    "class": "string",
    "confidence": "Float64",
    "entry_direction": "string",
    "exit_direction": "string",
}
