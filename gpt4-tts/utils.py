from typing import Optional

from pytube import YouTube


def download_youtube_video(url: str, output_path: Optional[str] = None) -> None:
    """
    Downloads a YouTube video from the given URL and saves it to the specified output path or the current directory.

    Args:
        url: The URL of the YouTube video to download.
        output_path: The path where the downloaded video will be saved. If None, the video will be saved to the current
        directory.

    Returns:
        None
    """
    yt = YouTube(url)

    video_stream = (
        yt.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
    )

    if output_path:
        video_stream.download(output_path)
        print(f"Video successfully downloaded to {output_path}")
    else:
        video_stream.download()
        print("Video successfully downloaded to the current directory")

