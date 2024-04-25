import os

from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, VideoFileClip

from src.main.utils.files_manipulation import get_path_of_files


class CreateVideo:
    @staticmethod
    def with_audio(
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

    @staticmethod
    def from_frames_with_video_audio(
            frames_dir: str,
            video_path: str,
            save_path: str,
            video_name: str,
            fps: float,
            file_type: str = ".jpg"
    ) -> None:
        """
        This method takes the frames from a directory and a videos audio and transforms them into a new video with audio.

        :param frames_dir: The full path to the directory that has the frames.
        :param video_path: The full path to the video file.
        :param save_path: The full path to where to save the video
        :param video_name: The name of the video to save (e.g., processed_frames_video)
        :param fps: Number of frames per second (should be most of the time 30)
        :param file_type: Can be .jpg, .png, .jpeg etc.
        :return: Void.
        """
        frames_paths = get_path_of_files(directory=frames_dir, file_type=file_type)
        frames = [ImageClip(frame_path).set_duration(1 / fps) for frame_path in frames_paths]
        video_file = VideoFileClip(video_path)
        audio = video_file.audio
        video = concatenate_videoclips(frames, method="compose")
        video = video.set_audio(audio)
        video.write_videofile(os.path.join(save_path, f"{video_name}.mp4"), fps=fps)

    @staticmethod
    def no_audio(
            frames_dir: str,
            save_path: str,
            video_name: str,
            fps: float,
            file_type: str = ".jpg"
    ) -> None:
        """
        This method takes the frames from a directory and an audio file and glue them together to form a .mp4 video.

        :param frames_dir: The full path to the directory that has the frames.
        :param save_path: The full path to where to save the video
        :param video_name: The name of the video to save (e.g., processed_frames_video)
        :param fps: Number of frames per second (should be most of the time 30)
        :param file_type: Can be .jpg, .png, .jpeg etc.
        :return: Void.
        """
        frames_paths = get_path_of_files(directory=frames_dir, file_type=file_type)
        frames = [ImageClip(frame_path).set_duration(1 / fps) for frame_path in frames_paths]
        video = concatenate_videoclips(frames, method="compose")
        video.write_videofile(os.path.join(save_path, f"{video_name}.mp4"), fps=fps)


if __name__ == '__main__':
    from src.main.utils.path_builder import Paths

    CreateVideo.with_audio(
        frames_dir=Paths.BAD_APPLE_FRAMES_DIR,
        audio_path=Paths.BAD_APPLE_AUDIO_PATH,
        save_path=Paths.BAD_APPLE_PROCESSED_VIDEO_DIR,
        video_name="bad-apple-radish",
        fps=30,
        file_type=".jpg"
    )

    CreateVideo.no_audio(
        frames_dir="./",
        save_path="./",
        video_name="bad-apple-test",
        fps=30,
        file_type=".jpg"
    )
