from pathlib import Path
from typing import Tuple, Optional
import cv2 as cv
import numpy as np
import warnings


class VideoHandler:
    """wrapper around opencv VideoCapture object that allows easy access to
    various camera properties as well as traversing the video and reading/writing
    frames.
    """

    def __init__(self, input_path: str, output_dir: str | None = None):
        """initializes video properties

        Arguments
        ---------
        input_path (str):
            the absolute path to the video source
        output_path (str | None):
            the absolute path to the video output DIRECTORY (VideoHandler will
            handle creating the filename for the video in there)
        """

        self._input_path = Path(input_path)
        self._video = cv.VideoCapture(input_path)
        if self._video is None or not self._video.isOpened():
            self._video.release()
            raise Exception(f"Could not open video at {input_path}")

        self._width = int(self._video.get(cv.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._video.get(cv.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._video.get(cv.CAP_PROP_FPS)
        self._total_frames = int(self._video.get(cv.CAP_PROP_FRAME_COUNT))
        self._current_frame = int(self._video.get(cv.CAP_PROP_POS_FRAMES))

        self._video_writer: Optional[cv.VideoWriter] = None
        self._output_dir = output_dir
        self._output_video_file = self._input_path.stem + "_out.mp4"

    def __iter__(self):
        while True:
            self._current_frame += 1
            ret, frame = self._video.read()
            if not ret or frame is None:
                break
            yield frame

        # cleanup
        self.cleanup()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._width, self._height

    @property
    def frame_count(self) -> int:
        return self._total_frames

    @property
    def current_frame(self) -> int:
        return self._current_frame

    def get_time_from_frame(self, frame_index: int) -> float:
        """computes the length of time from the start of the video to the specified
        frame in seconds
        """
        return frame_index / self._fps

    def get_frame(self, frame_index: int) -> np.ndarray | None:
        """grabs the frame at the specified frame index from the video"""
        if frame_index > self._total_frames or frame_index < 0:
            warnings.warn("invalid frame index")
            return None

        self._video.set(cv.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self._video.read()
        if ret:
            self._video.set(cv.CAP_PROP_POS_FRAMES, self._current_frame)
            return frame
        else:
            self._video.set(cv.CAP_PROP_POS_FRAMES, self._current_frame)
            warnings.warn("could not read frame")
            return None

    def set_current_frame(self, frame_index: int):
        """changes the current frame of the video"""
        if frame_index > self._total_frames or frame_index < 0:
            warnings.warn("invalid frame index")
        else:
            self._current_frame = frame_index
            self._video.set(cv.CAP_PROP_POS_FRAMES, frame_index)

    def show(self, img: np.ndarray, wait_time: int = 10, label: str = "output") -> int:
        """shows the image in a gui and returns the key pressed on it"""
        cv.imshow(label, img)
        return cv.waitKey(wait_time)

    def write_frame_to_video(self, frame: np.ndarray):
        """writes a frame to the output video"""
        if self._video_writer is None:
            # initial output setup happens here in case user changes resolution
            if self._output_dir:
                if not Path(self._output_dir).exists():
                    print("\nOutput directory does not exist, creating it ...\n")
                    Path(self._output_dir).mkdir(parents=True)
            else:
                # no output directory was specified, write to current directory
                self._output_dir = Path.cwd()

            outpath = str(Path(self._output_dir) / self._output_video_file)
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv.VideoWriter(
                outpath,
                fourcc=fourcc,
                fps=self._fps,
                frameSize=(frame.shape[1], frame.shape[0]),
            )

        self._video_writer.write(frame)

    # def write_frame_to_image(self, frame: np.ndarray, ext: str = "jpeg"):
    #     """writes a frame to an image file"""
    #     if ext not in ["png", "jpeg"]:
    #         warnings.warn("invalid file extension, writing as jpeg")
    #         ext = "jpeg"

    #     # TODO: Implement this functionality

    def get_output_path(self) -> Path | None:
        if self._output_dir:
            return Path(self._output_dir) / self._output_video_file
        else:
            return None

    def cleanup(self):
        if self._video_writer is not None:
            self._video_writer.release()
            print(f"Output video file saved to: {self.get_output_path()}")
        self._video.release()
        cv.destroyAllWindows()
