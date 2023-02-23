import os
import cv2
import numpy as np

from src.main.utils.path_builder import Paths


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
    # Resize the source image to match the size of the target image
    source_image = cv2.resize(source_image, target_image.shape[:2][::-1])

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


def apply_texture_on_frames():
    ...


if __name__ == '__main__':
    s_image = cv2.imread(Paths.RADISHES_PATH)
    t_image = cv2.imread(os.path.join(Paths.BAD_APPLE_FRAMES_DIR, "bad-apple_154.jpg"))
    tt_image = texture_transfer_min_error(source_image=s_image, target_image=t_image, block_size=4, overlap_width=1)
    # Save the texture-transferred image
    cv2.imshow("result", tt_image)
    cv2.waitKey(0)
    cv2.imwrite("texture_transferred_image.jpg", tt_image)
