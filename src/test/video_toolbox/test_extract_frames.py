import os
import unittest
from pathlib import Path

import cv2

from src.main.utils.files_manipulation import create_dir_if_doesnt_exist
from src.main.video_toolbox.extract_frames import FrameGenerator


class TestFrameGenerator(unittest.TestCase):
    def setUp(self):
        base_test_path = Path(__file__).resolve().parent.parent
        self.video_path = os.path.join(base_test_path, "resources", "video", "bad-apple-test.mp4")
        self.downscale_dim = (240, 180)
        self.upscale_dim = (960, 720)
        self.total_frames = 220
        self.fps = 30
        self.file_type = ".jpg"
        self.processed_path = os.path.join(base_test_path, "resources", "processed")
        create_dir_if_doesnt_exist(self.processed_path)

    def tearDown(self):
        if os.path.exists(self.processed_path):
            for file in os.listdir(self.processed_path):
                file_path = os.path.join(self.processed_path, file)
                os.remove(file_path)
            os.rmdir(self.processed_path)
        cv2.destroyAllWindows()

    def test_generate_all_frames(self):
        test_cases = [
            self.downscale_dim, self.upscale_dim
        ]
        for expected_width, expected_height in test_cases:
            with self.subTest(expected_width=expected_width, expected_height=expected_height):
                FrameGenerator.generate_all_frames(
                    self.video_path, self.processed_path, self.file_type, (expected_width, expected_height)
                )
                actual_frames_num = len(os.listdir(self.processed_path))
                self.assertEqual(self.total_frames, actual_frames_num)

                for frame_path in os.listdir(self.processed_path):
                    frame = cv2.imread(os.path.join(self.processed_path, frame_path))
                    actual_height, actual_width, _ = frame.shape
                    self.assertEqual(expected_width, actual_width)
                    self.assertEqual(expected_height, actual_height)

    def test_get_inter_based_on_resize(self):
        test_cases = [
            ((1920, 1080), (1280, 720), cv2.INTER_AREA),
            ((1280, 720), (1920, 1080), cv2.INTER_LANCZOS4),
            ((1920, 1080), (2500, 500), cv2.INTER_LINEAR)
        ]

        for video_dim, resize_dim, expected_inter in test_cases:
            with self.subTest(video_dim=video_dim, resize_dim=resize_dim, expected_inter=expected_inter):
                actual_inter = FrameGenerator.get_inter_based_on_resize(video_dim, resize_dim)
                self.assertEqual(actual_inter, expected_inter)

    def test_generate_frames(self):
        expected_frames_num = 4

        FrameGenerator.generate_frames(
            self.video_path, fps=2, processed_path=self.processed_path, file_type=self.file_type
        )
        actual_frames_num = len(os.listdir(self.processed_path))

        self.assertEqual(expected_frames_num, actual_frames_num)

    def test_get_frame(self):
        has_frame = FrameGenerator._FrameGenerator__get_frame(
            video_path=self.video_path,
            second=1.0,
            count=1,
            processed_path=self.processed_path,
            file_type=self.file_type
        )
        self.assertTrue(has_frame)

    def test_get_fps(self):
        actual_fps = FrameGenerator._FrameGenerator__get_fps(self.video_path)
        self.assertEqual(self.fps, actual_fps)


if __name__ == '__main__':
    unittest.main()
