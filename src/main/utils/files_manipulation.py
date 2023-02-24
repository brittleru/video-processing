import os
import re
import glob

from pathlib import PurePath

from src.main.utils.logging import Color
from src.main.utils.path_builder import Paths


def create_dir_if_doesnt_exist(path_to_dir: str) -> bool:
    """
    This will create the directories if they are not already existing.

    :param path_to_dir: The full path for the new directory.
    :return: A binary value that represents if the directories were created or not.
    """
    dir_exists = os.path.exists(path_to_dir)

    if not dir_exists:
        os.makedirs(path_to_dir, exist_ok=False)
        print(f"{Color.GREEN}Successfully created {Color.BOLD}'{PurePath(path_to_dir).name}'{Color.RESET}{Color.GREEN} "
              f"directory.{Color.RESET}")
        return True

    print(f"{Color.YELLOW}Directory {Color.BOLD}'{PurePath(path_to_dir).name}'{Color.RESET}{Color.YELLOW}"
          f" already exists.{Color.RESET}")
    return False


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


def purge_files_in_dir(path_to_dir: str) -> bool:
    """
    This will delete all the files from a given directory, the condition is for the directory to exist and to have
    at least one file in the given directory.

    :param path_to_dir: The given directory to delete the files.
    :return: A binary value that represents if the files were deleted or not.
    """
    dir_exists = os.path.exists(path_to_dir)

    if dir_exists:
        files = glob.glob(os.path.join(path_to_dir, "*"))
        if len(files) > 0:
            for f in files:
                os.remove(f)
            print(f"{Color.GREEN}Successfully deleted all contents in "
                  f"{Color.BOLD}'{PurePath(path_to_dir).name}'{Color.RESET}{Color.GREEN} directory.{Color.RESET}")
            return True
        else:
            print(f"{Color.YELLOW}Directory {Color.BOLD}'{PurePath(path_to_dir).name}'{Color.RESET}{Color.YELLOW}"
                  f" is empty.{Color.RESET}")
            return False

    print(f"{Color.RED}Directory {Color.BOLD}'{PurePath(path_to_dir).name}'{Color.RESET}{Color.RED}"
          f" doesn't exist.{Color.RESET}")
    return False


def generate_project_structure() -> None:
    """
    Method to create all the directories needed for the project, some files you need to add yourself.

    :return: Void.
    """
    create_dir_if_doesnt_exist(Paths.RESOURCES_DIR)
    create_dir_if_doesnt_exist(Paths.AUDIO_DIR)
    create_dir_if_doesnt_exist(Paths.VIDEO_DIR)
    create_dir_if_doesnt_exist(Paths.PROCESSED_DIR)
    create_dir_if_doesnt_exist(Paths.BAD_APPLE_PROCESSED_DIR)
    create_dir_if_doesnt_exist(Paths.BAD_APPLE_FRAMES_DIR)
    create_dir_if_doesnt_exist(Paths.BAD_APPLE_TEXTURE_TRANSFER_DIR)
    create_dir_if_doesnt_exist(Paths.BAD_APPLE_RADISH_DIR)
    create_dir_if_doesnt_exist(Paths.BAD_APPLE_RICE_DIR)
    create_dir_if_doesnt_exist(Paths.BAD_APPLE_SPAGHETTI_DIR)
    create_dir_if_doesnt_exist(Paths.BAD_APPLE_PROCESSED_VIDEO_DIR)

    create_dir_if_doesnt_exist(Paths.TEXTURES_DIR)


if __name__ == '__main__':
    generate_project_structure()
