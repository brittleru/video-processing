import os
from pathlib import Path
from typing import Final


class Paths:
    BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent.parent
    RESOURCES_DIR: Final[str] = os.path.join(BASE_DIR, "resources")

    AUDIO_DIR: Final[str] = os.path.join(RESOURCES_DIR, "audio")
    BAD_APPLE_AUDIO_PATH: Final[str] = os.path.join(AUDIO_DIR, "bad-apple-audio.mp3")

    VIDEO_DIR: Final[str] = os.path.join(RESOURCES_DIR, "video")
    BAD_APPLE_VIDEO_PATH: Final[str] = os.path.join(VIDEO_DIR, "bad-apple.mp4")

    PROCESSED_DIR: Final[str] = os.path.join(RESOURCES_DIR, "processed")
    BAD_APPLE_PROCESSED_DIR: Final[str] = os.path.join(PROCESSED_DIR, "bad-apple")

    TEXTURES_DIR: Final[str] = os.path.join(RESOURCES_DIR, "textures")
    RICE_PATH: Final[str] = os.path.join(TEXTURES_DIR, "rice.jpg")
    RADISHES_PATH: Final[str] = os.path.join(TEXTURES_DIR, "radishes.jpg")
    SPAGHETTI_PATH: Final[str] = os.path.join(TEXTURES_DIR, "spaghetti.jpeg")


if __name__ == '__main__':
    print(Paths.BASE_DIR)
    print(Paths.RESOURCES_DIR)

    print(Paths.AUDIO_DIR)
    print(Paths.BAD_APPLE_AUDIO_PATH)

    print(Paths.VIDEO_DIR)
    print(Paths.BAD_APPLE_VIDEO_PATH)

    print(Paths.PROCESSED_DIR)
    print(Paths.BAD_APPLE_PROCESSED_DIR)

