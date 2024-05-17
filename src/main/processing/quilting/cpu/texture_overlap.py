import concurrent.futures
import os
from multiprocessing import cpu_count
from time import time
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.main.utils.consts import ColorBGR
from src.main.utils.display import readable_time
from src.main.utils.files_manipulation import get_path_of_files
from src.main.utils.logging import Color
from src.main.utils.path_builder import Paths
from src.main.video_toolbox.extract_frames import FrameGenerator


class TextureOverlapping:
    def __init__(self, source_image: np.ndarray, target_size: Tuple[int, int], block_size: int = 4,
                 overlap_width: int = 1):
        """
        :param source_image: The image from where to "steal" the texture.
        :param target_size: The size of the image to apply texture to (w, h).
        :param block_size: Size of the patches of the source image, to obtain the desired level of granularity in the
                           texture transfer. For example if the blocks are too large then the texture transfer may be
                           too
                           coarse, if they are too small then the result may be too detailed.
        :param overlap_width: The padding of the block size.
        """
        self.block_size = block_size
        self.overlap_width = overlap_width
        self.target_size = target_size
        self.interpolation = FrameGenerator.get_inter_based_on_resize(source_image.shape[:2], target_size)
        self.source_image = cv2.resize(source_image, target_size, interpolation=self.interpolation)

    @staticmethod
    def visualize_process(vis_target: np.ndarray, vis_result: np.ndarray, block_left: int,
                          block_top: int, block_right: int, block_bottom: int):
        cv2.rectangle(vis_target, (block_left, block_top), (block_right, block_bottom), ColorBGR.GREEN, 2)
        cv2.rectangle(vis_result, (block_left, block_top), (block_right, block_bottom), ColorBGR.GREEN, 2)

        cv2.imshow('Target Block', vis_target)
        cv2.imshow('Intermediate Result', vis_result)
        cv2.waitKey(100)

    @staticmethod
    def find_min_cost_path(error_block: np.ndarray) -> np.ndarray:
        min_cost_path = np.zeros_like(error_block, dtype=np.int32)
        min_cost_path[0, 0] = error_block[0, 0]

        for i in range(1, min_cost_path.shape[0]):
            min_cost_path[i, 0] = min_cost_path[i - 1, 0] + error_block[i, 0]

        for j in range(1, min_cost_path.shape[1]):
            min_cost_path[0, j] = min_cost_path[0, j - 1] + error_block[0, j]

        for i in range(1, min_cost_path.shape[0]):
            for j in range(1, min_cost_path.shape[1]):
                min_cost_path[i, j] = error_block[i, j] + min(
                    min_cost_path[i - 1, j], min_cost_path[i, j - 1], min_cost_path[i - 1, j - 1]
                )

        return min_cost_path

    @staticmethod
    def trace_min_cost_path(min_cost_path: np.ndarray, block_left: int, block_top: int) -> tuple[list[int], list[int]]:
        path_x = []
        path_y = []
        i = min_cost_path.shape[0] - 1
        j = min_cost_path.shape[1] - 1

        while i > 0 or j > 0:
            path_x.append(j + block_left)
            path_y.append(i + block_top)
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                if min_cost_path[i - 1, j] <= min_cost_path[i, j - 1] and \
                        min_cost_path[i - 1, j] <= min_cost_path[i - 1, j - 1]:
                    i -= 1
                elif min_cost_path[i, j - 1] <= min_cost_path[i - 1, j] and \
                        min_cost_path[i, j - 1] <= min_cost_path[i - 1, j - 1]:
                    j -= 1
                else:
                    i -= 1
                    j -= 1
        path_x.append(block_left)
        path_y.append(block_top)

        return path_x, path_y

    def blend_blocks(self, source_block: np.ndarray, target_block: np.ndarray, path_x: list[int], path_y: list[int],
                     block_top: int, block_left: int) -> np.ndarray:
        blend_mask = np.zeros((target_block.shape[0], target_block.shape[1]), dtype=np.float32)

        for i in range(len(path_x)):
            blend_mask[
                max(0, path_y[i] - self.overlap_width - block_top):
                min(target_block.shape[0], path_y[i] + self.overlap_width - block_top),
                max(0, path_x[i] - self.overlap_width - block_left):
                min(target_block.shape[1], path_x[i] + self.overlap_width - block_left)
            ] = 1
        blend_mask = cv2.GaussianBlur(blend_mask, (2 * self.overlap_width + 1, 2 * self.overlap_width + 1), 0)
        blend_mask = np.clip(blend_mask, 0, 1)

        blended_block = (
                source_block.astype(np.float32) * blend_mask[:, :, np.newaxis] +
                target_block.astype(np.float32) * (1 - blend_mask[:, :, np.newaxis])
        )

        return blended_block.astype(np.uint8)

    def texture_transfer_min_error(self, target_image: np.ndarray, is_visualized: bool = True) -> np.ndarray:
        """
        Method to apply the texture transfer between two images. The source image will be resized to the match the target
        image by inferring an interpolation based on. The error surface will be computed between the source and target
        images, and it will be divided into overlapping blocks. Then it will parse the image block by block to find the
        minimum error block that is similar with the target block.

        :param target_image: The image to apply texture to.
        :param is_visualized: If its true it will perform an OpenCV visualization.
        :return: The new image with the applied texture transfer.
        """
        error_surface = np.sum(np.square(
            self.source_image.astype(np.float32) - target_image.astype(np.float32)
        ), axis=2)
        num_blocks_x = int(np.ceil(
            (self.source_image.shape[1] - self.overlap_width) / (self.block_size - self.overlap_width)
        ))
        num_blocks_y = int(np.ceil(
            (self.source_image.shape[0] - self.overlap_width) / (self.block_size - self.overlap_width)
        ))

        texture_transferred_image = np.zeros_like(self.source_image)

        for block_y in range(num_blocks_y):
            for block_x in range(num_blocks_x):
                block_left = block_x * (self.block_size - self.overlap_width)
                block_top = block_y * (self.block_size - self.overlap_width)
                block_right = min(block_left + self.block_size, self.source_image.shape[1])
                block_bottom = min(block_top + self.block_size, self.source_image.shape[0])

                error_block = error_surface[block_top:block_bottom, block_left:block_right]
                min_cost_path = self.find_min_cost_path(error_block)
                path_x, path_y = self.trace_min_cost_path(min_cost_path, block_left, block_top)

                texture_block = self.source_image[block_top:block_bottom, block_left:block_right]
                target_block = target_image[block_top:block_bottom, block_left:block_right]
                blended_block = self.blend_blocks(texture_block, target_block, path_x, path_y, block_top, block_left)
                texture_transferred_image[block_top:block_bottom, block_left:block_right] = blended_block

                if is_visualized:
                    vis_target = target_image.copy()
                    vis_result = texture_transferred_image.copy()
                    self.visualize_process(
                        vis_target, vis_result, block_left, block_top, block_right, block_bottom
                    )

        if is_visualized:
            cv2.destroyAllWindows()
        return texture_transferred_image

    def resize_source_image(self, target_image: np.ndarray):
        target_size = target_image.shape[:2][::-1]
        interpolation = FrameGenerator.get_inter_based_on_resize(self.source_image.shape[:2], target_size)
        self.source_image = cv2.resize(self.source_image, target_size, interpolation=interpolation)

    def process_frame(self, _frame_path: str, save_dir: str, is_visualized: bool = False) -> str:
        """
        This method will take a frame, transform it into a numpy array and apply texture transfer on it then the
        processed frame will be saved. It's a helper method for multiprocessing, since doing this work sequentially
        would take a lot of time.

        :param _frame_path: Full path of the frame to process.
        :param save_dir: Where to save the processed frame.
        :param is_visualized: If its true it will perform an OpenCV visualization.
        :return: The file name, needed for multiprocessing.
        """
        _frame_name = os.path.basename(_frame_path)
        _frame = cv2.imread(_frame_path)
        _frame = self.texture_transfer_min_error(target_image=_frame, is_visualized=is_visualized)
        cv2.imwrite(os.path.join(save_dir, _frame_name), _frame)

        return _frame_name

    def apply_texture_on_frames(
            self,
            frames_dir: str,
            save_dir: str,
            file_type: str = ".jpg",
            is_visualized: bool = False
    ) -> None:
        """
        This method applies texture transfer on all the given frames with multiprocessing, on the number of the cores
        of you machine CPU minus 2 (so you can use your machine while the files are processed).

        :param frames_dir: The full path to the directory that has the frames to process.
        :param save_dir: The full path to the directory to save the processed frames.
        :param file_type: It can be .jpg, .png, .jpeg, etc.
        :param is_visualized: If its true it will perform an OpenCV visualization.
        :return: Void.
        """
        frames_paths = get_path_of_files(directory=frames_dir, file_type=file_type)
        frames_paths_size = len(frames_paths)

        if len(frames_paths) == 0:
            raise RuntimeError(f"Directory doesn't contain any '{file_type}' frames... '{frames_dir}' is empty")
        temp_frame = cv2.imread(frames_paths[0])
        self.resize_source_image(temp_frame)

        print(f"{Color.BOLD}Texture transferring frames...{Color.RESET}")
        print(f"Total of {Color.BOLD}({frames_paths_size}){Color.RESET} frames.")

        start_time = time()
        cpus = cpu_count() - 2
        if cpus <= 0:
            cpus = 1
        with concurrent.futures.ProcessPoolExecutor(cpus) as executor:
            with tqdm(total=frames_paths_size) as progress_bar:
                futures = {}
                for index, frame_path in enumerate(frames_paths):
                    future = executor.submit(
                        self.process_frame, frame_path, save_dir, is_visualized
                    )
                    futures[future] = index
                results = [None] * frames_paths_size
                for future in concurrent.futures.as_completed(futures):
                    index = futures[future]
                    results[index] = future.result()
                    progress_bar.update(1)
            res = [result for result in results]

        end_time = time()
        print(f"Finished ({Color.BOLD}{len(res)}{Color.RESET}) frames. {readable_time(start_time, end_time)}")


if __name__ == '__main__':
    s_image = cv2.imread(Paths.RICE_PATH)
    t_image = cv2.imread(Paths.EMINESCU_PATH)
    textured = TextureOverlapping(
        source_image=s_image, target_size=t_image.shape[:2][::-1], block_size=4, overlap_width=1
    )
    tt_image_name = textured.process_frame(_frame_path=Paths.EMINESCU_PATH, save_dir="../../", is_visualized=False)

    textured.apply_texture_on_frames(
        frames_dir=Paths.BAD_APPLE_FRAMES_DIR,
        save_dir=Paths.PROCESSED_DIR,
    )
