import os
import re
import glob


def get_path_of_files(directory: str, file_type: str = ".jpg") -> list:
    """
    Function to get a list of paths for every file in a given directory.
    Since the paths will be strings they won't be ordered (e.g., after
    the image 1.jpeg the next image will be 10.jpeg) they were sorted by using a regex.

    :param directory: This must be a string with the path of the directory the videos or the images are in.
    :param file_type: The type of file encoding format e.g., .mp4, .avi, .wmv, .jpg, .png etc.
    :return: A list with paths to the videos or images.
    """
    file_paths = []
    for file_path in glob.glob(os.path.join(directory, f"*{file_type}")):
        file_paths.append(file_path)

    file_paths.sort(key=lambda f: int(re.sub("\\D", "", f)))
    return file_paths
