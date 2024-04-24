import os
from pathlib import Path
from typing import Final


class Paths:
    """Contains all the paths needed for computation"""

    BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent.parent
    RESOURCES_DIR: Final[str] = os.path.join(BASE_DIR, "resources")

    AUDIO_DIR: Final[str] = os.path.join(RESOURCES_DIR, "audio")
    BAD_APPLE_AUDIO_PATH: Final[str] = os.path.join(AUDIO_DIR, "bad-apple-audio.mp3")

    VIDEO_DIR: Final[str] = os.path.join(RESOURCES_DIR, "video")
    BAD_APPLE_VIDEO_PATH: Final[str] = os.path.join(VIDEO_DIR, "bad-apple.mp4")
    BAD_APPLE_MMD_VIDEO_PATH: Final[str] = os.path.join(VIDEO_DIR, "bad_apple_mmd_model.mp4")

    PROCESSED_DIR: Final[str] = os.path.join(RESOURCES_DIR, "processed")
    BAD_APPLE_PROCESSED_DIR: Final[str] = os.path.join(PROCESSED_DIR, "bad-apple")
    BAD_APPLE_FRAMES_DIR: Final[str] = os.path.join(BAD_APPLE_PROCESSED_DIR, "frames")
    BAD_APPLE_MMD_FRAMES_DIR: Final[str] = os.path.join(BAD_APPLE_PROCESSED_DIR, "mmd-frames")
    BAD_APPLE_TEXTURE_TRANSFER_DIR: Final[str] = os.path.join(BAD_APPLE_PROCESSED_DIR, "texture-transfer")

    BAD_APPLE_PROCESSED_VIDEO_DIR: Final[str] = os.path.join(BAD_APPLE_PROCESSED_DIR, "video")

    TEXTURES_DIR: Final[str] = os.path.join(RESOURCES_DIR, "textures")
    RICE_PATH: Final[str] = os.path.join(TEXTURES_DIR, "rice.jpg")
    RADISHES_PATH: Final[str] = os.path.join(TEXTURES_DIR, "radishes.jpg")
    SPAGHETTI_PATH: Final[str] = os.path.join(TEXTURES_DIR, "spaghetti.jpeg")

    INPUTS_DIR: Final[str] = os.path.join(RESOURCES_DIR, "inputs")
    EMINESCU_PATH: Final[str] = os.path.join(INPUTS_DIR, "eminescu.jpg")

    LOGS_DIR: Final[str] = os.path.join(BASE_DIR, "logs")


if __name__ == '__main__':
    print(Paths.BASE_DIR)
    print(Paths.RESOURCES_DIR)

    print(Paths.AUDIO_DIR)
    print(Paths.BAD_APPLE_AUDIO_PATH)

    print(Paths.VIDEO_DIR)
    print(Paths.BAD_APPLE_VIDEO_PATH)
    print(Paths.BAD_APPLE_MMD_VIDEO_PATH)

    print(Paths.PROCESSED_DIR)
    print(Paths.BAD_APPLE_PROCESSED_DIR)
    print(Paths.BAD_APPLE_FRAMES_DIR)
    print(Paths.BAD_APPLE_MMD_FRAMES_DIR)
    print(Paths.BAD_APPLE_TEXTURE_TRANSFER_DIR)

    print(Paths.BAD_APPLE_PROCESSED_VIDEO_DIR)

    print(Paths.TEXTURES_DIR)
    print(Paths.RICE_PATH)
    print(Paths.RADISHES_PATH)
    print(Paths.SPAGHETTI_PATH)

    print(Paths.INPUTS_DIR)
    print(Paths.EMINESCU_PATH)

    print(Paths.LOGS_DIR)
