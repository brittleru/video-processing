import os
from time import time
from typing import Tuple

import cv2

from src.main.utils.logging import Color
from src.main.utils.path_builder import Paths
from src.main.utils.progress import ProgressBar


class FrameGenerator:
    @staticmethod
    def generate_all_frames(
            video: str,
            processed_path: str = Paths.BAD_APPLE_FRAMES_DIR,
            file_type: str = ".jpg",
            resize_dim: Tuple[int, int] | None = None
    ) -> None:
        """
        Will will take all the frames from a video and save them on a given path.

        :param video: Video path.
        :param processed_path: Where to save the frames.
        :param file_type: Image file type to save, e.g., can be .jpg, .png, .jpeg etc.
        :param resize_dim: If this is given then it will resize the frames to the given dimension by checking if it
                           needs to use an interpolation to shrink or enlarge an image.
        :return: Void
        """
        video_capture = cv2.VideoCapture(video)
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length_digits_len = len(str(video_length))
        video_name = os.path.splitext(os.path.basename(video))[0]
        video_dim = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        interpolation = FrameGenerator.get_inter_based_on_resize(video_dim, resize_dim)

        print(f"{Color.BOLD}Processing video frames...{Color.RESET}")
        print(f"Video has {Color.BOLD}{video_length}{Color.RESET} total frames.")
        count = 0
        start_time = time()
        while video_capture.isOpened():
            iter_time = time()
            success, image = video_capture.read()
            if not success:
                continue

            save_processed_path = os.path.join(processed_path, f"{video_name}_{count}{file_type}")
            if interpolation is not None:
                image = cv2.resize(image, resize_dim, interpolation=interpolation)
            cv2.imwrite(save_processed_path, image)
            ProgressBar.run(video_length, count + 1, iter_time, display_iter_width=video_length_digits_len)
            count = count + 1

            if count > (video_length - 1):
                end_time = time()
                print(f"Finished ({Color.BOLD}{count}{Color.RESET}) frames. Time took: {end_time - start_time:.02f}s")
                video_capture.release()
                break
        video_capture.release()
        cv2.destroyAllWindows()

    @staticmethod
    def get_inter_based_on_resize(video_dim: Tuple[int, int], resize_dim: Tuple[int, int] | None) -> int | None:
        """
        Will infer the interpolation needed for the resize based on what size the video has.

        :param video_dim: Video width and height (w, h) as a tuple.
        :param resize_dim: The new resize width and height (w, h) as a tuple. If its values are lower than the video
                           ones then it will choose a down-scaling interpolation (cv2.INTER_AREA) and if the values are
                           higher than the video ones it will perform an up-scaling interpolation (cv2.INTER_LANCZOS4).
                           If you want a non-uniform scaling the cv2.INTER_LINEAR will be used.
        :return: The interpolation type.
        """
        if resize_dim is None:
            return None

        video_width, video_height = video_dim
        resize_width, resize_height = resize_dim
        if video_width > resize_width and video_height > resize_height:
            return cv2.INTER_AREA
        elif video_width < resize_width and video_height < resize_height:
            return cv2.INTER_LANCZOS4
        return cv2.INTER_LINEAR

    @staticmethod
    def generate_frames(
            video: str,
            fps: float = None,
            processed_path: str = Paths.BAD_APPLE_FRAMES_DIR,
            file_type: str = ".jpg"
    ) -> None:
        """
        Generate all the frames from a video with the number of frames per second. E.g., if you only want two frames
        per second or 30. If you want to take all the frames from the video better use
        :ref: `FrameGenerator.generate_all_frames`.

        :param video: Path to the wanted video.
        :param fps: How many frames to take per second. (i.e., if you don't want to take all the frames). If the
                    parameter is None then it will be converted to the frames of the video.
        :param processed_path: Path to save the frames.
        :param file_type: Image file type to save, e.g., can be .jpg, .png, .jpeg etc.
        :return: Void.
        """
        if fps is None:
            fps = 1 / FrameGenerator.__get_fps(video_path=video)
        second = 0
        count = 1
        print(f"{Color.BOLD}Processing video frames at {fps} FPS...{Color.RESET}")
        success = FrameGenerator.__get_frame(video, second, count, processed_path, file_type)

        while success:
            count += 1
            second += fps
            second = round(second, 2)
            success = FrameGenerator.__get_frame(video, second, count, processed_path, file_type)

        print(f"{Color.GREEN}Finished ({count - 1}) frames. Time took: {Color.RESET}")

    @staticmethod
    def __get_frame(
            video_path: str,
            second: float,
            count: int = 1,
            processed_path: str = Paths.BAD_APPLE_FRAMES_DIR,
            file_type: str = ".jpg"
    ) -> bool:
        """
        Function to get the specific frame of the video and save it as an image.jpg.

        :param video_path: Path to the wanted video.
        :param second: The second from which to take the frames.
        :param count: Image number.
        :param processed_path: Path to save the frames.
        :param file_type: Image file type to save, e.g., can be .jpg, .png, .jpeg etc.
        :return: A binary value that express that there is a frame.
        """
        video_capture = cv2.VideoCapture(video_path)
        video_capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        has_frames, image = video_capture.read()

        video_name = os.path.splitext(os.path.basename(video_path))[0]

        if has_frames:
            save_processed_path = os.path.join(processed_path, f"{video_name}_{count}{file_type}")
            cv2.imwrite(save_processed_path, image)

        return has_frames

    @staticmethod
    def __get_fps(video_path: str) -> float:
        """
        Will return the number of FPS from a given video.

        :param video_path: Path to the wanted video.
        :return: The frames per second.
        """
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        cv2.destroyAllWindows()

        return fps


if __name__ == '__main__':
    # FrameGenerator.generate_frames(
    #     video=Paths.BAD_APPLE_VIDEO_PATH, fps=1, processed_path="./", file_type=".jpg"
    # )
    FrameGenerator.generate_all_frames(
        video="./bad-apple-test.mp4", processed_path="./"
    )
