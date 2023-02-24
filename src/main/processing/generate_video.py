import os

from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

from src.main.utils.files_manipulation import get_path_of_files


def create_video_with_audio(
        frames_dir: str,
        audio_path: str,
        save_path: str,
        video_name: str,
        fps: float,
        file_type: str = ".jpg"
) -> None:
    """
    This method takes the frames from a directory and an audio file and glue them together to form a .mp4 video.

    :param frames_dir: The full path to the directory that has the frames.
    :param audio_path: The full path to the audio file.
    :param save_path: The full path to where to save the video
    :param video_name: The name of the video to save (e.g., processed_frames_video)
    :param fps: Number of frames per second (should be most of the time 30)
    :param file_type: Can be .jpg, .png, .jpeg etc.
    :return: Void.
    """
    frames_paths = get_path_of_files(directory=frames_dir, file_type=file_type)
    frames = [ImageClip(frame_path).set_duration(1 / fps) for frame_path in frames_paths]
    audio = AudioFileClip(audio_path)
    video = concatenate_videoclips(frames, method="compose")
    video = video.set_audio(audio)
    video.write_videofile(os.path.join(save_path, f"{video_name}.mp4"), fps=fps)


if __name__ == '__main__':
    from src.main.utils.path_builder import Paths

    # create_video_with_audio(
    #     frames_path=Paths.BAD_APPLE_RADISH_DIR,
    #     audio_path=Paths.BAD_APPLE_AUDIO_PATH,
    #     save_path=Paths.BAD_APPLE_PROCESSED_VIDEO_DIR,
    #     video_name="bad-apple-radish",
    #     fps=30,
    #     file_type=".jpg"
    # )
    create_video_with_audio(
        frames_dir=Paths.BAD_APPLE_RICE_DIR,
        audio_path=Paths.BAD_APPLE_AUDIO_PATH,
        save_path=Paths.BAD_APPLE_PROCESSED_VIDEO_DIR,
        video_name="bad-apple-rice",
        fps=30,
        file_type=".jpg"
    )
    # create_video_with_audio(
    #     frames_path=Paths.BAD_APPLE_SPAGHETTI_DIR,
    #     audio_path=Paths.BAD_APPLE_AUDIO_PATH,
    #     save_path=Paths.BAD_APPLE_PROCESSED_VIDEO_DIR,
    #     video_name="bad-apple-spaghetti",
    #     fps=30,
    #     file_type=".jpg"
    # )
