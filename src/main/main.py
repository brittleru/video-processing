from src.main.utils.files_manipulation import generate_project_structure
from src.main.processing.extract_frames import FrameGenerator
from src.main.utils.path_builder import Paths


def setup(
        video_path: str = Paths.BAD_APPLE_VIDEO_PATH,
        save_processed_imgs: str = Paths.BAD_APPLE_PROCESSED_DIR,
        file_type: str = ".jpg"
):
    generate_project_structure()
    FrameGenerator.generate_all_frames(video=video_path, processed_path=save_processed_imgs, file_type=file_type)

