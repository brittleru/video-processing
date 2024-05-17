import os
import concurrent.futures
import math
import logging
from multiprocessing import cpu_count
from time import time

import cv2
import numpy as np
from numba import jit
from tqdm import tqdm

from src.main.processing.texture_synthesis import l2_norm, min_err_boundary_cut, unravel_index
from src.main.utils.consts import QuiltingTypes
from src.main.utils.display import readable_time
from src.main.utils.files_manipulation import get_path_of_files
from src.main.utils.logging import Color
from src.main.utils.path_builder import Paths


logging.basicConfig(
    filename="../../../logs/texture_transfer_multiprocessing2.log",
    format='%(levelname)s - %(asctime)s - %(message)s',
    # filemode='w',
    encoding="utf-8"
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


@jit(nopython=True)
def sample_correlation(
        texture: np.ndarray,
        gray_texture: np.ndarray,
        block_size: int,
        gray_target: np.ndarray,
        vertical_coord: int,
        horizontal_coord: int
) -> np.ndarray:
    """
    Finds the patch in the original texture that matches the best with the sample centered at the target pixel with
    respect with the correlation values.

    :param texture: The texture where we take the sample from as a numpy array.
    :param gray_texture: The image where each pixel value is the correlation value between a sample centered on that
                         pixel and the samples around the target pixel.
    :param block_size: The size of sample from the texture.
    :param gray_target: The correlation values of the sample centered at the target pixel
    :param vertical_coord: Row index of the target pixel
    :param horizontal_coord: Column index of the target Pixel
    :return: The sample from the texture image that corresponds to the best sample.
    """
    height_texture, width_texture, _ = texture.shape
    errors = np.zeros((height_texture - block_size, width_texture - block_size))

    gray_target_sample = gray_target[
                         vertical_coord:vertical_coord + block_size, horizontal_coord:horizontal_coord + block_size
                         ]
    height_target, width_target = gray_target_sample.shape

    for i in range(height_texture - block_size):
        for j in range(width_texture - block_size):
            gray_texture_sample = gray_texture[i:i + height_target, j:j + width_target]
            e = gray_texture_sample - gray_target_sample
            errors[i, j] = np.sum(e ** 2)

    best_height, best_width = unravel_index(np.argmin(errors), errors.shape)
    return texture[best_height:best_height + height_target, best_width:best_width + width_target]


@jit(nopython=True)
def sample_best_overlapping(
        texture: np.ndarray, gray_texture: np.ndarray, block_size: int, overlap_width: int, gray_target: np.ndarray,
        current_image: np.ndarray, vertical_coord: int, horizontal_coord: int, alpha: float = 0.1, level: int = 0
) -> np.ndarray:
    """
    Finds the sample in the original image that best matches the patch centered in the target pixel in terms of both
    correlation values and L2 norm errors, using the sample overlap and previously reconstructed samples from the
    image.

    :param texture: The texture where we take the sample from as a numpy array.
    :param gray_texture: The image where each pixel value is the correlation value between a sample centered on that
                         pixel and the samples around the target pixel.
    :param block_size: The size of sample from the texture.
    :param overlap_width: The overlapping region between samples.
    :param gray_target: The correlation values of the sample centered at the target pixel
    :param current_image: A numpy array representing the reconstructed image so far.
    :param vertical_coord: Row index of the target pixel
    :param horizontal_coord: Column index of the target Pixel
    :param alpha: The overlap error relative to the correlation error.
    :param level: The level of the image pyramid to handle the patch overlap.
    :return: The sample from the image that corresponds to the best sample in the texture.
    """
    height_texture, width_texture, _ = texture.shape
    errors = np.zeros((height_texture - block_size, width_texture - block_size))

    gray_target_sample = gray_target[
                         vertical_coord:vertical_coord + block_size,
                         horizontal_coord:horizontal_coord + block_size
                         ]
    height_target, width_target = gray_target_sample.shape

    for i in range(height_texture - block_size):
        for j in range(width_texture - block_size):
            patch = texture[i:i + height_target, j:j + width_target]
            l2_norm_error = l2_norm(patch, block_size, overlap_width, current_image, vertical_coord, horizontal_coord)

            gray_texture_sample = gray_texture[i:i + height_target, j:j + width_target]
            gray_error = np.sum((gray_texture_sample - gray_target_sample) ** 2)

            last_error = 0
            if level > 0:
                last_error = patch[overlap_width:, overlap_width:] - \
                             current_image[
                             vertical_coord + overlap_width:vertical_coord + block_size,
                             horizontal_coord + overlap_width:horizontal_coord + block_size
                             ]
                last_error = np.sum(last_error ** 2)

            errors[i, j] = alpha * (l2_norm_error + last_error) + (1 - alpha) * gray_error

    best_height, best_width = unravel_index(np.argmin(errors), errors.shape)
    return texture[best_height:best_height + height_target, best_width:best_width + width_target]


@jit(nopython=True)
def compute_transfer(
        output: np.ndarray, texture: np.ndarray, gray_texture: np.ndarray, gray_target: np.ndarray,
        num_sample_height: int, num_sample_width: int, block_size: int, overlap_width: int,
        alpha: float, level: int, sample_type: str,
):
    for i in range(num_sample_height):
        for j in range(num_sample_width):
            vertical_coord = i * (block_size - overlap_width)
            horizontal_coord = j * (block_size - overlap_width)
            if i == 0 and j == 0 or sample_type == "sample_correlation":
                sample = sample_correlation(
                    texture, gray_texture, block_size, gray_target, vertical_coord, horizontal_coord
                )
            elif sample_type == "sample_overlap":
                sample = sample_best_overlapping(
                    texture, gray_texture, block_size, overlap_width, gray_target, output, vertical_coord,
                    horizontal_coord
                )
            elif sample_type == "minimum_error":
                sample = sample_best_overlapping(
                    texture, gray_texture, block_size, overlap_width, gray_target,
                    output, vertical_coord, horizontal_coord, alpha, level
                )
                sample = min_err_boundary_cut(output, sample, overlap_width, vertical_coord, horizontal_coord)
            # else:
            #     raise NotImplementedError(f"Quilting type: {sample_type} is not implemented yet...")

            output[vertical_coord:vertical_coord + block_size, horizontal_coord:horizontal_coord + block_size] = sample

    return output


def transfer_texture(
        texture: np.ndarray, target: np.ndarray, block_size: int, alpha: float = 0.1, level: int = 0,
        prev_image: np.ndarray = None, sample_type: str = "minimum_error"
) -> np.ndarray:
    """
    Apply texture transfer between a source (texture) and a target image. It creates an empty image as output,
    if it's the first level of the transfer then it's initialized as zeros, otherwise it uses the previous image.

    :param texture: The texture image that is wanted to be used as transfer (i.e., source).
    :param target: The image where we want to apply the new texture.
    :param block_size: The size of sample from the texture.
    :param alpha: The overlap error relative to the correlation error.
    :param level: The level of the image pyramid to handle the patch overlap.
    :param prev_image: The last image from the result of the transfer, if not given then the output image is
                       initialized with zeros.
    :param sample_type: Which type of sampling you want, i.e., 'sample_correlation', 'sample_overlap' or 'minimum_error'
    :return: The new image with the texture applied from source to target.
    """
    gray_texture = cv2.cvtColor(texture, cv2.COLOR_RGB2GRAY)
    gray_target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    texture = texture[:, :, :3].astype(np.float64) / 255.0
    target = target[:, :, :3].astype(np.float64) / 255.0
    height_target, width_target, _ = target.shape
    overlap_width = block_size // 6
    num_sample_height = math.ceil((height_target - block_size) / (block_size - overlap_width)) + 1 or 1
    num_sample_width = math.ceil((width_target - block_size) / (block_size - overlap_width)) + 1 or 1

    if level == 0:
        output = np.zeros_like(target)
    else:
        output = prev_image

    return compute_transfer(
        output, texture, gray_texture, gray_target, num_sample_height,
        num_sample_width, block_size, overlap_width, alpha, level, sample_type
    )


def texture_transfer_pyramid(
        texture: np.ndarray, target: np.ndarray, block_size: int,
        num_iterations: int, sample_type: str = QuiltingTypes.MINIMUM_ERROR
) -> np.ndarray:
    """
    Applies texture transfer by a given iteration, a higher number of iterations means a better texture transferred
    image.

    :param texture: The texture image that is wanted to be used as transfer (i.e., source).
    :param target: The image where we want to apply the new texture.
    :param block_size: The size of sample from the texture.
    :param num_iterations: The number of iterations to apply texture transfer, in each iteration the last generated
                           image is used to generate a better transfer.
    :param sample_type: Which type of sampling you want, i.e., 'sample_correlation', 'sample_overlap' or 'minimum_error'
    :return: The new image with the texture applied from source to target.
    """
    logger.info("Iteration 0...")
    output = transfer_texture(texture, target, block_size, sample_type=sample_type)
    for i in range(1, num_iterations):
        logger.info(f"Iteration {i}...")
        alpha = (0.8 * (i - 1) / (num_iterations - 1)) + 0.1
        block_size = block_size * 2 ** i // 3 ** i
        output = transfer_texture(
            texture, target, block_size, alpha=alpha, level=i, prev_image=output, sample_type=sample_type
        )

    return (output * 255).astype(np.uint8)


def run_texture_transfer(source_img_path, target_img_path, result_name, num_iterations: int = 2):
    source = cv2.imread(source_img_path)
    target = cv2.imread(target_img_path)

    print(f"Applying texture transfer on {result_name}...")
    start_time = time()
    result = texture_transfer_pyramid(source, target, 20, num_iterations)
    end_time = time()
    readable_time(start_time, end_time)

    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(result_name, result)


def run_texture_transfer_multiprocessing(
        source_img_path: str, target_img_path: str, result_name: str, save_path: str, num_iterations: int = 2
) -> str:
    logger.info(f"Applying texture transfer on {result_name}")
    source = cv2.imread(source_img_path)
    target = cv2.imread(target_img_path)
    result = texture_transfer_pyramid(source, target, 20, num_iterations)
    cv2.imwrite(os.path.join(save_path, result_name), result)
    logger.info(f"Saving {result_name} to {save_path}")

    return result_name


def run_texture_transfer_batch_multiprocessing(
        frames_dir: str,
        source_img_path: str = Paths.RICE_PATH,
        save_path: str = Paths.BAD_APPLE_PROCESSED_DIR,
        num_iterations: int = 2,
        file_type: str = ".jpg",
        where_to_slice_paths_arr: int = 0
):
    target_paths = get_path_of_files(directory=frames_dir, file_type=file_type)[where_to_slice_paths_arr:]
    target_paths_size = len(target_paths)
    logger.info(f"Starting to process {target_paths_size} frames.")
    logger.info(f"Applying {os.path.basename(source_img_path)} texture.")
    start_time = time()
    with concurrent.futures.ProcessPoolExecutor(cpu_count() - 2) as executor:
        with tqdm(total=target_paths_size) as progress_bar:
            futures = {}
            for index, frame_path in enumerate(target_paths):
                future = executor.submit(
                    run_texture_transfer_multiprocessing,
                    source_img_path, frame_path, os.path.basename(frame_path), save_path, num_iterations
                )
                futures[future] = index
            results = [None] * target_paths_size
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                logger.info(f"Process number {index} done")
                results[index] = future.result()
                progress_bar.update(1)
        res = [result for result in results]

    end_time = time()
    logger.info(f"Finished ({len(res)}) frames. {readable_time(start_time, end_time, False, False)}")
    print(f"Finished ({Color.BOLD}{len(res)}{Color.RESET}) frames. {readable_time(start_time, end_time)}")


if __name__ == '__main__':

    run_texture_transfer(
        Paths.RICE_PATH, os.path.join(Paths.BAD_APPLE_MMD_FRAMES_DIR, "bad_apple_mmd_model_1.jpg"),
        "ba-mmd-1.png",
        num_iterations=2
    )

    # # TODO: run this for a smaller number of frames, processed 1183 frames (including the 0th frame)
    # run_texture_transfer_batch_multiprocessing(
    #     frames_dir=Paths.BAD_APPLE_MMD_FRAMES_DIR,
    #     source_img_path=Paths.RICE_PATH,
    #     save_path=Paths.BAD_APPLE_RICE_DIR,
    #     num_iterations=2,
    #     where_to_slice_paths_arr=5807
    # )
