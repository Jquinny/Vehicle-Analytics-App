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


class RequiredInputFoundMenu(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Required Input Found")
        self.setGeometry(100, 100, 300, 150)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        instructions = (
            "File containing ROI polygon and direction coordinates found for this\n"
            "camera view, would you like to use them? Press yes to use existing information.\n"
            "Press no to continue to ROI and direction coordinate selection."
        )
        label = QLabel(instructions, self)
        layout.addWidget(label)

        self.yes_button = QPushButton("Yes", self)
        self.yes_button.clicked.connect(self.on_yes)
        layout.addWidget(self.yes_button)

        self.no_button = QPushButton("No", self)
        self.no_button.clicked.connect(self.on_no)
        layout.addWidget(self.no_button)

        self.central_widget.setLayout(layout)

        self.use_existing = False

    def on_yes(self):
        self.use_existing = True
        self.close()

    def on_no(self):
        self.close()

    def closeEvent(self, event):
        event.accept()


class InvalidPolygonHandler(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Invalid Polygon")
        self.setGeometry(100, 100, 300, 150)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        instructions = (
            "Invalid Region of Interest.\n\n"
            "Make sure no polygon sides cross at the end and that there are at least 3 points.\n\n"
            "Press the retry button to retry polygon creation, or close this window to continue "
            "without a region of interest."
        )
        label = QLabel(instructions, self)
        layout.addWidget(label)

        self.retry_button = QPushButton("Retry", self)
        self.retry_button.clicked.connect(self.on_retry)
        layout.addWidget(self.retry_button)

        self.central_widget.setLayout(layout)

        self.quit = True

    def on_retry(self):
        self.quit = False
        self.close()

    def closeEvent(self, event):
        event.accept()


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


def check_existing_user_input() -> bool:
    """if .json file containing required user input such as ROI and coordinates
    exists in the parent directory of the video being processed, pops up a gui
    for the user to select if they want to use the information in that file or not

    Returns
    -------
    bool:
        whether to use existing user input or not
    """

    app = QApplication(sys.argv)
    dialog = RequiredInputFoundMenu()
    dialog.show()
    app.exec_()

    return dialog.use_existing


def handle_invalid_polygon() -> bool:
    """opens up a small gui for the user to choose to retry polygon input or
    continue without ROI

    Returns
    -------
    bool:
        whether or not to quit the ROI selection process
    """
    app = QApplication(sys.argv)
    invalid_handler = InvalidPolygonHandler()
    invalid_handler.show()
    app.exec_()

    return invalid_handler.quit


def get_coordinate_input() -> str | None:
    """opens up a small gui for the user to enter the direction associated with
    a coordinate

    Returns
    -------
    str | None:
        the text associated with the coordinate, or None if user didn't want
        to keep the coordinate
    """
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
        "Press enter to show final result. Press enter again to confirm, or press "
        "another key to continue."
    )
    draw_text(resized_frame, instructions, Point(x=10, y=10))

    data = {"roi": Poly(), "original": resized_frame, "copy": resized_frame.copy()}

    # grab ROI from user
    cv.namedWindow(window_title)
    cv.setMouseCallback(window_title, get_roi_points, data)

    while True:
        cv.imshow(window_title, data["copy"])
        if cv.waitKey(10) == ord("\r"):
            # show final results
            data["copy"] = data["original"].copy()
            draw_roi(data["copy"], data["roi"], True)
            cv.imshow(window_title, data["copy"])

            if len(data["roi"]) < 3 or not data["roi"].is_valid():
                data["roi"].clear_coords()
                if handle_invalid_polygon():
                    cv.destroyWindow(window_title)
                    return None
                data["copy"] = data["original"].copy()
                continue

            if cv.waitKey(0) == ord("\r"):
                cv.destroyWindow(window_title)
                break
            else:
                # user wants to continue, reset to showing unclosed polygon
                data["copy"] = data["original"].copy()
                draw_roi(data["copy"], data["roi"], False)

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
