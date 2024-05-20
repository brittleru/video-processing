import numpy as np
from numba import jit


@jit(nopython=True)
def copy_to(target: np.ndarray, source: np.ndarray, mask: np.ndarray):
    """
    Iterate over each index of the arrays and copy the value where the mask has presence.
    It also checks if all images have the same size.
    """
    if target.shape != source.shape or target.shape != mask.shape:
        raise ValueError("Shape mismatch: dst, src, and where must have the same shape")

    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            for k in range(target.shape[2]):
                if mask[i][j][k]:
                    target[i][j][k] = source[i][j][k]


@jit(nopython=True)
def unravel_index(index, shape):
    """
    Convert a flat index into a tuple of multidimensional indices.

    :param index: A flat index into an array.
    :param shape: The shape of the array.
    :return: A tuple of multidimensional indices corresponding to the given flat index.
    """
    n_dim = len(shape)
    indices = []
    for i in range(n_dim - 1, -1, -1):
        indices.append(index % shape[i])
        index //= shape[i]

    indices = indices[::-1]
    return indices


@jit(nopython=True)
def random_sample(image: np.ndarray, block_size: int) -> np.ndarray:
    """
    Generates a block (patch) of the given size at random starting coordinates in the image.

    :param image: The input image representing the texture as a numpy array.
    :param block_size: The size of the random sample.
    :return: The random sample.
    """
    sample_h = np.random.randint(image.shape[0] - block_size)
    sample_w = np.random.randint(image.shape[1] - block_size)

    return image[sample_h:sample_h + block_size, sample_w:sample_w + block_size]


@jit(nopython=True)
def l2_norm(
        sample: np.ndarray,
        block_size: int,
        overlap_width: int,
        current_image: np.ndarray,
        vertical_coord: int,
        horizontal_coord: int
) -> float:
    """
    Computes the L2 norm error between a sample and the corresponding region of an output image, taking into account the
    amount of overlap between the block and its neighbors.

    :param sample: A sample block from an image that is of a given size.
    :param block_size: The size determining the sample.
    :param overlap_width: The size of the overlap between blocks.
    :param current_image: A numpy array representing the image.
    :param vertical_coord: Row index of the top-left corner of the block in the image.
    :param horizontal_coord: Column index of the top-left corner of the block in the image.
    :return: The L2 norm error.
    """
    l2_err = 0
    if horizontal_coord > 0:
        width_left = sample[:, :overlap_width] - \
                     current_image[
                     vertical_coord:vertical_coord + block_size,
                     horizontal_coord:horizontal_coord + overlap_width
                     ]
        l2_err += np.sum(width_left ** 2)
    if vertical_coord > 0:
        height_upwards = sample[:overlap_width, :] - \
                         current_image[
                         vertical_coord:vertical_coord + overlap_width,
                         horizontal_coord:horizontal_coord + block_size
                         ]
        l2_err += np.sum(height_upwards ** 2)
    if horizontal_coord > 0 and vertical_coord > 0:
        corner = sample[:overlap_width, :overlap_width] - \
                 current_image[
                 vertical_coord:vertical_coord + overlap_width, horizontal_coord:horizontal_coord + overlap_width
                 ]
        l2_err -= np.sum(corner ** 2)

    return l2_err
