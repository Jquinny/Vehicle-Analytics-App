"""TODO: add type hints"""

import json
import cv2 as cv
import numpy as np
import warnings

# user entered information
ROI = []
COORDINATES = {}


def draw_roi(img: np.ndarray, close: bool = False):
    """Draw ROI polygon onto image"""
    global ROI

    np_roi = np.array(ROI)
    for point in np_roi:
        cv.circle(img, point, 3, (0, 255, 0), 3)
    cv.polylines(img, [np_roi], close, (0, 255, 0), 3)

    return


def get_points(event, x, y, flags, images):
    """Get list of points for ROI from user mouse input"""
    global ROI

    if event == cv.EVENT_LBUTTONDOWN:
        ROI.append((x, y))
        draw_roi(images["copy"])

    if event == cv.EVENT_RBUTTONDOWN:
        if ROI:
            ROI.pop()
            images["copy"] = images["original"].copy()  # erasing lines to redraw
            draw_roi(images["copy"])

    return


def get_roi(frame: np.ndarray, roi_file: str = None):
    global ROI

    # get roi from user mouse input if no input roi json file
    if roi_file is None:
        images = {"original": frame, "copy": frame.copy()}

        # grab ROI from user
        cv.namedWindow("first frame")
        cv.setMouseCallback("first frame", get_points, images)
        print("\nPress enter to connect final points.")
        print(
            "Press enter again if satisfied. Press any other key to continue editing.\n"
        )
        while True:
            cv.imshow("first frame", images["copy"])
            key = cv.waitKey(10)
            if key == ord("\r"):  # show full polygon when user presses enter
                images["copy"] = images["original"].copy()
                draw_roi(images["copy"], True)
                cv.imshow("first frame", images["copy"])
                key = cv.waitKey(0)
                if key == ord("\r"):
                    cv.destroyWindow("first frame")
                    break

    #     # save ROI to json if user wants
    #     ans = input("Would you like to save ROI to a json file? (y/n): ")
    #     while ans not in ("y", "n"):
    #         ans = input("Not a valid input, try again: ")
    #     if ans == "y":
    #         roi_filename = input("Specify filename for roi (no file extension): ")
    #         with open(roi_filename + ".json", "w") as json_file:
    #             json.dump(ROI, json_file)
    # else:
    #     with open(roi_file) as f:
    #         ROI = json.load(f)

    return ROI


def get_coordinates(frame: np.ndarray):
    global COORDINATES
    warnings.warn("\nget_coordinates(frame) is not implemented yet")
    return COORDINATES


def draw_coordinates(frame: np.ndarray):
    global COORDINATES
    warnings.warn("\ndraw_coordinates(frame) is ")
