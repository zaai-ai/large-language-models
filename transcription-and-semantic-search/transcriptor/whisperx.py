import whisper
from whisperx import align, load_align_model

from base.config import Config


class WhisperX(Config):
    """Transcription Model"""

    def __init__(self):
        super().__init__()
        self.model = whisper.load_model(
            self.config["transcriptor"]["model"], self.config["transcriptor"]["device"]
        )

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribes the audio
        Args:
            audio_path (str): path to .wav file
        Returns:
            result_aligned (dict): dictionary with segments and metadata
        """

        result = self.model.transcribe(audio_path)
        model_a, metadata = load_align_model(
            language_code=result["language"],
            device=self.config["transcriptor"]["device"],
        )
        result_aligned = align(
            result["segments"],
            model_a,
            metadata,
            audio_path,
            self.config["transcriptor"]["device"],
        )

        return result_aligned
