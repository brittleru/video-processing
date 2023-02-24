import os
import cv2
import numpy as np
import concurrent.futures

from time import time
from tqdm import tqdm
from multiprocessing import cpu_count

from src.main.utils.logging import Color
from src.main.utils.path_builder import Paths
from src.main.utils.display import readable_time
from src.main.utils.files_manipulation import get_path_of_files


def texture_transfer_min_error(
        source_image: np.ndarray,
        target_image: np.ndarray,
        block_size: int = 4,
        overlap_width: int = 1
) -> np.ndarray:
    """
    Method to apply the texture transfer between two images.

    :param source_image: The image from where to "steal" the texture
    :param target_image: The image to apply texture to.
    :param block_size: Size of the patches of the source image, to obtain the desired level of granularity in the
                       texture transfer. For example if the blocks are too large then the texture transfer may be too
                       coarse, if they are too small then the result may be too detailed.
    :param overlap_width: The padding of the block size.
    :return: The new image with the applied texture transfer.
    """
    source_h, source_w = source_image.shape[:2]
    target_h, target_w = target_image.shape[:2]
    if source_h > target_h or source_w > source_w:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    # Resize the source image to match the size of the target image
    source_image = cv2.resize(source_image, target_image.shape[:2][::-1], interpolation=interp)

    # Compute the error surface between the source and target images
    error_surface = np.sum(np.square(source_image.astype(np.float32) - target_image.astype(np.float32)), axis=2)

    # Divide the error surface into overlapping blocks
    num_blocks_x = int(np.ceil((source_image.shape[1] - overlap_width) / (block_size - overlap_width)))
    num_blocks_y = int(np.ceil((source_image.shape[0] - overlap_width) / (block_size - overlap_width)))

    texture_transferred_image = np.zeros_like(source_image)

    for block_y in range(num_blocks_y):
        for block_x in range(num_blocks_x):
            # Compute the bounds of the current block
            block_left = block_x * (block_size - overlap_width)
            block_top = block_y * (block_size - overlap_width)
            block_right = min(block_left + block_size, source_image.shape[1])
            block_bottom = min(block_top + block_size, source_image.shape[0])

            # Compute the minimum cost path through the error surface for the current block
            error_block = error_surface[block_top:block_bottom, block_left:block_right]
            min_cost_path = np.zeros_like(error_block, dtype=np.int32)
            min_cost_path[0, 0] = error_block[0, 0]

            for i in range(1, min_cost_path.shape[0]):
                min_cost_path[i, 0] = min_cost_path[i - 1, 0] + error_block[i, 0]

            for j in range(1, min_cost_path.shape[1]):
                min_cost_path[0, j] = min_cost_path[0, j - 1] + error_block[0, j]

            for i in range(1, min_cost_path.shape[0]):
                for j in range(1, min_cost_path.shape[1]):
                    min_cost_path[i, j] = error_block[i, j] + min(
                        min_cost_path[i - 1, j], min_cost_path[i, j - 1], min_cost_path[i - 1, j - 1])

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

            # Apply the texture transfer by blending
            texture_block = source_image[block_top:block_bottom, block_left:block_right]
            target_block = target_image[block_top:block_bottom, block_left:block_right]
            blend_mask = np.zeros_like(error_block, dtype=np.float32)

            for i in range(len(path_x)):
                blend_mask[
                    max(0, path_y[i] - overlap_width - block_top):
                    min(block_bottom - block_top, path_y[i] + overlap_width - block_top),
                    max(0, path_x[i] - overlap_width - block_left):
                    min(block_right - block_left, path_x[i] + overlap_width - block_left)
                ] = 1
            blend_mask = cv2.GaussianBlur(blend_mask, (2 * overlap_width + 1, 2 * overlap_width + 1), 0)
            blend_mask = np.clip(blend_mask, 0, 1)
            texture_block = texture_block.astype(np.float32) * blend_mask[:, :, np.newaxis] + target_block.astype(
                np.float32) * (1 - blend_mask[:, :, np.newaxis])
            texture_transferred_image[block_top:block_bottom, block_left:block_right] = texture_block.astype(np.uint8)

    return texture_transferred_image


def process_frame(
        _frame_path: str,
        source_image: np.ndarray,
        textured_frames_dir: str,
        block_size: int = 4,
        overlap_width: int = 1
) -> str:
    """
    This method will take a frame, transform it into a numpy array and apply texture transfer on it then the processed
    frame will be saved. It's a helper method for multiprocessing, since doing this work sequentially would take a lot
    of time.

    :param _frame_path: Full path of the frame to process.
    :param source_image: Full path of the texture to apply
    :param textured_frames_dir: Where to save the processed frame.
    :param block_size: Size of the patches of the source image, to obtain the desired level of granularity in the
                       texture transfer. For example if the blocks are too large then the texture transfer may be too
                       coarse, if they are too small then the result may be too detailed.
    :param overlap_width: The padding of the block size.
    :return: The file name, needed for multiprocessing.
    """
    _frame_name = os.path.basename(_frame_path)
    _frame = cv2.imread(_frame_path)
    _frame = texture_transfer_min_error(
        source_image=source_image, target_image=_frame, block_size=block_size, overlap_width=overlap_width
    )
    cv2.imwrite(os.path.join(textured_frames_dir, _frame_name), _frame)
    return _frame_name


def apply_texture_on_frames(
        frames_dir: str,
        textured_frames_dir: str,
        source_texture_path: str,
        block_size: int = 4,
        overlap_width: int = 1,
        file_type: str = ".jpg"
) -> None:
    """
    This method applies texture transfer on all the given frames with multiprocessing, on the number of the cores
    of you machine CPU minus 2 (so you can use your machine while the files are processed).

    :param frames_dir: The full path to the directory that has the frames to process.
    :param textured_frames_dir: The full path to the directory to save the processed frames.
    :param source_texture_path: The full path to the texture image to apply.
    :param block_size: Size of the patches of the source image, to obtain the desired level of granularity in the
                       texture transfer. For example if the blocks are too large then the texture transfer may be too
                       coarse, if they are too small then the result may be too detailed.
    :param overlap_width: The padding of the block size.
    :param file_type: It can be .jpg, .png, .jpeg, etc.
    :return: Void.
    """
    frames_paths = get_path_of_files(directory=frames_dir, file_type=file_type)
    source_image = cv2.imread(source_texture_path)
    frames_paths_size = len(frames_paths)

    print(f"{Color.BOLD}Texture transferring frames...{Color.RESET}")
    print(f"Total of {Color.BOLD}({frames_paths_size}){Color.RESET} frames.")

    start_time = time()
    with concurrent.futures.ProcessPoolExecutor(cpu_count() - 2) as executor:
        with tqdm(total=frames_paths_size) as progress_bar:
            futures = {}
            for index, frame_path in enumerate(frames_paths):
                future = executor.submit(
                    process_frame, frame_path, source_image, textured_frames_dir, block_size, overlap_width
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
    t_image = cv2.imread(os.path.join(Paths.BAD_APPLE_FRAMES_DIR, "bad-apple_154.jpg"))
    tt_image = texture_transfer_min_error(source_image=s_image, target_image=t_image, block_size=4, overlap_width=2)
    # Save the texture-transferred image
    cv2.imshow("result", tt_image)
    cv2.waitKey(0)
    cv2.imwrite("texture_transferred_image.jpg", tt_image)
    apply_texture_on_frames(
        Paths.BAD_APPLE_FRAMES_DIR, Paths.BAD_APPLE_RADISH_DIR, Paths.RADISHES_PATH,
        block_size=4,
        overlap_width=1
    )

