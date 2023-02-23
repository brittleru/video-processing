import os
from pathlib import PurePath
from src.main.utils.logging import Color
from src.main.utils.path_builder import Paths


def create_dir_if_doesnt_exist(path_to_dir: str) -> bool:
    dir_exists = os.path.exists(path_to_dir)

    if not dir_exists:
        os.makedirs(path_to_dir, exist_ok=False)
        print(f"{Color.GREEN}Successfully created {Color.BOLD}'{PurePath(path_to_dir).name}'{Color.RESET}{Color.GREEN} "
              f"directory.{Color.RESET}")
        return True

    print(f"{Color.YELLOW}Directory {Color.BOLD}'{PurePath(path_to_dir).name}'{Color.RESET}{Color.YELLOW}"
          f" already exists.{Color.RESET}")
    return False


if __name__ == '__main__':
    create_dir_if_doesnt_exist(Paths.RESOURCES_DIR)
    create_dir_if_doesnt_exist(Paths.AUDIO_DIR)
    create_dir_if_doesnt_exist(Paths.VIDEO_DIR)
    create_dir_if_doesnt_exist(Paths.PROCESSED_DIR)
    create_dir_if_doesnt_exist(Paths.BAD_APPLE_PROCESSED_DIR)
    create_dir_if_doesnt_exist(Paths.TEXTURES_DIR)
