import heapq
import os.path
from typing import List

import cv2
import numpy as np
from numba import jit

from src.main.processing.quilting.cpu.utils import l2_norm, unravel_index, copy_to, random_sample
from src.main.utils.consts import QuiltingTypes


@jit(nopython=True)
def neigh_blocks_constrained_by_overlap(
        current_image: np.ndarray,
        texture: np.ndarray,
        block_size: int,
        overlap_width: int,
        vertical_coord: int,
        horizontal_coord: int
) -> np.ndarray:
    """
    This method finds the best matching patch in the original texture for a given block in the output image,
    using the overlap with neighboring blocks. It uses L2 Norm error to compute the error between the patch
    and the corresponding region of the image, it returns the sample with the minimal error.

    :param texture: A numpy array representing the texture image.
    :param block_size: The size determining the sample.
    :param overlap_width: The size of the overlap between blocks.
    :param current_image: A numpy array representing the image.
    :param vertical_coord: Row index of the top-left corner of the block in the image.
    :param horizontal_coord: Column index of the top-left corner of the block in the image.
    :return: The best matching patch in the texture, sliced from the texture.
    """
    texture_height, texture_width, _ = texture.shape
    l2_norms = np.zeros((texture_height - block_size, texture_width - block_size))

    for i in range(texture_height - block_size):
        for j in range(texture_width - block_size):
            sample = texture[i:i + block_size, j:j + block_size]
            l2_norm_error = l2_norm(sample, block_size, overlap_width, current_image, vertical_coord, horizontal_coord)
            l2_norms[i, j] = l2_norm_error

    best_height, best_width = unravel_index(np.argmin(l2_norms), l2_norms.shape)
    return texture[best_height:best_height + block_size, best_width:best_width + block_size]


@jit(nopython=True)
def min_cost(errors: np.ndarray) -> List[int]:
    """
    Dijkstra's algorithm to find the minimum cost path in a matrix of errors. Finds the minimum cost on a vertical
    path from the top row to the bottom row of the matrix. This method also keeps track of the seen pixels to
    avoid visiting the same pixel twice.

    :param errors: A numpy array representing the cost of pixels at a given position.
    :return: The path with the lowest total cost from the top tow to the bottom row, which is a list of indices
             corresponding to the column positions of the pixels.
    """
    priority_queue = [(error, [i]) for i, error in enumerate(errors[0])]
    heapq.heapify(priority_queue)
    height, width = errors.shape
    seen_pixels = set()

    while priority_queue:
        err, path = heapq.heappop(priority_queue)
        temp_depth = len(path)
        temp_index = path[-1]

        if temp_depth == height:
            return path

        for delta in (-1, 0, 1):
            next_index = temp_index + delta
            if 0 <= next_index < width:
                if (temp_depth, next_index) not in seen_pixels:
                    total_err = err + errors[temp_depth, next_index]
                    heapq.heappush(priority_queue, (total_err, path + [next_index]))
                    seen_pixels.add((temp_depth, next_index))


@jit(nopython=True)
def min_err_boundary_cut(
        current_image: np.ndarray,
        sample: np.ndarray,
        overlap_width: int,
        vertical_coord: int,
        horizontal_coord: int
) -> np.ndarray:
    """
    Implementation of minimum error boundary cut algorithm.

    :param current_image: A numpy array representing the image.
    :param sample: A numpy array representing the sample.
    :param overlap_width: The size of the overlap between blocks.
    :param vertical_coord: Row index of the top-left corner of the block in the image.
    :param horizontal_coord: Column index of the top-left corner of the block in the image.
    :return: The new image with the minimum cut patch replaced on the current image.
    """
    sample = sample.copy()
    height_sample, width_sample, _ = sample.shape
    minimum_cut = np.zeros(sample.shape, dtype=np.bool_)

    if horizontal_coord > 0:
        horizontal = sample[:, :overlap_width] - \
                     current_image[
                     vertical_coord:vertical_coord + height_sample, horizontal_coord:horizontal_coord + overlap_width
                     ]
        horizontal_err = np.sum(horizontal ** 2, axis=2)
        for i, j in enumerate(min_cost(horizontal_err)):
            minimum_cut[i, :j] = True
    if vertical_coord > 0:
        vertical = sample[:overlap_width, :] - \
                   current_image[
                   vertical_coord:vertical_coord + overlap_width, horizontal_coord:horizontal_coord + width_sample
                   ]
        vertical_err = np.sum(vertical ** 2, axis=2)
        for j, i in enumerate(min_cost(vertical_err.T)):
            minimum_cut[:i, j] = True

    copy_to(
        sample,
        current_image[vertical_coord:vertical_coord + height_sample, horizontal_coord:horizontal_coord + width_sample],
        mask=minimum_cut
    )
    return sample


class ImageQuilting:
    def __init__(self, block_size: int = 36, num_block: int = 10, sample_type: str = QuiltingTypes.MINIMUM_ERROR):
        """
        :param block_size: The size of sample from the image.
        :param num_block: The number of blocks to fill the new generated image.
        :param sample_type: Which type of sampling you want, i.e., 'random_placement', 'neighboring_blocks',
                            'minimum_error'
        """
        self.block_size = block_size
        self.num_block = (num_block, num_block)
        self.sample_type = sample_type
        if self.sample_type not in [
            QuiltingTypes.RANDOM_PLACEMENT, QuiltingTypes.NEIGHBORING_BLOCKS, QuiltingTypes.MINIMUM_ERROR
        ]:
            raise NotImplementedError(f"Quilting type: {self.sample_type} is not implemented yet...")

    def quilt(self, image_path: str):
        """
        Implements the quilting algorithm to enhance an image with samples from it. It has three possible types of
        generation, those described in the Image Quilting for Texture Synthesis and Transfer paper, one to take random
        samples from the input image, one to use the neighboring blocks constrained by overlap and one that uses
        minimum error boundary cut method.

        :param image_path: Full path of the image to apply quilt.
        :return: The new quilted image.
        """
        texture = cv2.imread(image_path)
        texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
        texture = texture.astype(np.float64) / 255.0
        overlap_width = self.block_size // 6
        output_height = (self.num_block[0] * self.block_size) - (self.num_block[0] - 1) * overlap_width
        output_width = (self.num_block[1] * self.block_size) - (self.num_block[1] - 1) * overlap_width
        output = np.zeros((output_height, output_width, texture.shape[2]), dtype=np.float64)

        for i in range(self.num_block[0]):
            for j in range(self.num_block[1]):
                vertical_coord = i * (self.block_size - overlap_width)
                horizontal_coord = j * (self.block_size - overlap_width)
                if i == 0 and j == 0 or self.sample_type == QuiltingTypes.RANDOM_PLACEMENT:
                    sample = random_sample(texture, self.block_size)
                elif self.sample_type == QuiltingTypes.NEIGHBORING_BLOCKS:
                    sample = neigh_blocks_constrained_by_overlap(
                        output, texture, self.block_size, overlap_width, vertical_coord, horizontal_coord
                    )
                elif self.sample_type == QuiltingTypes.MINIMUM_ERROR:
                    sample = neigh_blocks_constrained_by_overlap(
                        output, texture, self.block_size, overlap_width, vertical_coord, horizontal_coord
                    )
                    sample = min_err_boundary_cut(output, sample, overlap_width, vertical_coord, horizontal_coord)
                else:
                    raise NotImplementedError(f"Quilting type: {self.sample_type} is not implemented yet...")

                output[
                    vertical_coord:vertical_coord + self.block_size,
                    horizontal_coord:horizontal_coord + self.block_size
                ] = sample

        output = cv2.cvtColor((output * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return output

    def run_quilt(self, image_path: str, save_path: str = "./"):
        texture = cv2.imread(image_path)

        start_time = time()
        output = self.quilt(image_path)
        end_time = time()
        print(f"Finished in: {end_time - start_time} seconds")

        cv2.imshow("Input", texture)
        cv2.imshow("Output", output)
        cv2.waitKey(0)
        cv2.imwrite(os.path.join(save_path, "output.png"), output)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    from src.main.utils.path_builder import Paths
    from time import time

    image_quilter = ImageQuilting(block_size=36, num_block=10, sample_type=QuiltingTypes.MINIMUM_ERROR)
    image_quilter.run_quilt(image_path=Paths.RICE_PATH, save_path="./")
