from typing import List, Dict, Tuple

import cv2 as cv
import numpy as np
import sys

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
)

from src.utils.geometry import Point, Poly
from src.utils.drawing import draw_coordinates, draw_roi, draw_text

WINDOW_SIZE = (1280, 720)


class CoordinateInputDialog(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Coordinate Selection")
        self.setGeometry(100, 100, 300, 150)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        instructions = "Enter Coordinate Direction"
        label = QLabel(instructions, self)
        layout.addWidget(label)

        self.text_input = QLineEdit(self)
        layout.addWidget(self.text_input)

        self.confirm_button = QPushButton("Confirm", self)
        self.confirm_button.clicked.connect(self.on_confirm)
        layout.addWidget(self.confirm_button)

        self.central_widget.setLayout(layout)

        self.result = None

    def on_confirm(self):
        self.result = self.text_input.text()
        self.close()

    def closeEvent(self, event):
        event.accept()


def get_coordinate_input():
    app = QApplication(sys.argv)
    dialog = CoordinateInputDialog()
    dialog.show()
    app.exec_()

    return dialog.result


def get_roi_points(event, x, y, flags, data: dict):
    """Get list of points for ROI from user mouse input"""
    if event == cv.EVENT_LBUTTONDOWN:
        data["roi"].push(Point(x=x, y=y))

    if event == cv.EVENT_RBUTTONDOWN:
        if data["roi"]:
            data["roi"].pop()
            data["copy"] = data["original"].copy()  # erasing lines to redraw

    draw_roi(data["copy"], data["roi"])

    return


def get_roi(frame: np.ndarray) -> Poly | None:
    window_title = "Region of Interest Extraction"
    window_scale_x = frame.shape[1] / WINDOW_SIZE[0]
    window_scale_y = frame.shape[0] / WINDOW_SIZE[1]
    resized_frame = cv.resize(frame, WINDOW_SIZE)

    instructions = (
        "Left-click to enter a polygon point. Right-click to remove. "
        "Press enter to show final result. Press enter again to confirm."
    )
    draw_text(resized_frame, instructions, Point(x=10, y=10))

    data = {"roi": Poly(), "original": resized_frame, "copy": resized_frame.copy()}

    # grab ROI from user
    cv.namedWindow(window_title)
    cv.setMouseCallback(window_title, get_roi_points, data)
    print("\nPress enter to connect final points.")
    print("Press enter again if satisfied. Press any other key to continue editing.\n")
    while True:
        cv.imshow(window_title, data["copy"])
        key = cv.waitKey(10)
        if key == ord("\r"):  # show full polygon when user presses enter
            data["copy"] = data["original"].copy()
            draw_roi(data["copy"], data["roi"], True)
            cv.imshow(window_title, data["copy"])
            key = cv.waitKey(0)
            if key == ord("\r"):
                cv.destroyWindow(window_title)
                break

    if not data["roi"].is_empty():
        # coordinate transform due to window scaling
        return Poly(
            [
                Point(x=pt.x * window_scale_x, y=pt.y * window_scale_y)
                for pt in data["roi"]
            ]
        )
    else:
        return None


def get_coordinate_points(event, x, y, flags, data: dict):
    """Get list of points for coordinate directions from user mouse input"""
    if event == cv.EVENT_LBUTTONDOWN:
        text = get_coordinate_input()
        if text:
            data["coordinates"][text] = Point(x=x, y=y)

    if event == cv.EVENT_RBUTTONDOWN:
        if data["coordinates"]:
            # remove the last entered coordinate
            data["coordinates"].popitem()
            data["copy"] = data["original"].copy()  # erasing coords to redraw

    draw_coordinates(data["copy"], data["coordinates"])

    return


def get_coordinates(frame: np.ndarray) -> Dict[str, Point]:
    window_title = "Direction Coordinate Placement"
    window_scale_x = frame.shape[1] / WINDOW_SIZE[0]
    window_scale_y = frame.shape[0] / WINDOW_SIZE[1]
    resized_frame = cv.resize(frame, WINDOW_SIZE)

    instructions = (
        "Left-click to enter a direction coordinate. Right-click to remove. "
        "Press enter to finish."
    )
    draw_text(resized_frame, instructions, Point(x=10, y=10))

    data = {
        "coordinates": {},
        "original": resized_frame,
        "copy": resized_frame.copy(),
    }

    # grab direction coordinates from user
    cv.namedWindow(window_title)
    cv.setMouseCallback(window_title, get_coordinate_points, data)
    print("\nPress enter to confirm.")
    while True:
        cv.imshow(window_title, data["copy"])
        key = cv.waitKey(10)
        if key == ord("\r"):
            cv.destroyWindow(window_title)
            break

    # coordinate transform due to window scaling
    return {
        dir: Point(x=pt.x * window_scale_x, y=pt.y * window_scale_y)
        for dir, pt in data["coordinates"].items()
    }
