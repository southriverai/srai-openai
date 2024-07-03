from typing import Literal, Optional

from openai import OpenAI
from srai_core.tools_env import get_string_from_env


class ClientOpenaiAudio:
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = get_string_from_env("OPENAI_API_KEY")
        self.client_openai = OpenAI(api_key=api_key)

    def get_default_model_id(self) -> str:
        return "whisper-1"

    def get_default_tts_model_id(self) -> str:
        return "tts-1"

    def transcription(
        self,
        path_file_audio: str,
    ) -> dict:
        # TODO add script
        model_id = self.get_default_model_id()
        with open(path_file_audio, "rb") as file:
            transcription = self.client_openai.audio.transcriptions.create(
                model=model_id, file=file, response_format="verbose_json", timestamp_granularities=["word"]
            )
        return transcription.model_dump()

    def text_to_speech_for_file(
        self,
        text: str,
        path_file_audio: str,
        *,
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy",
        response_format: Literal["mp3", "opus", "aac", "flac"] = "mp3",
        model_id: Optional[str] = None,
    ) -> None:
        if model_id is None:
            model_id = self.get_default_tts_model_id()
        speech = self.client_openai.audio.speech.create(
            model=model_id,
            voice=voice,
            response_format=response_format,
            input=text,
        )
        speech.write_to_file(path_file_audio)

    def text_to_speech_for_bytes(
        self,
        text: str,
        *,
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy",
        response_format: Literal["mp3", "opus", "aac", "flac"] = "mp3",
        model_id: Optional[str] = None,
    ) -> bytes:
        if model_id is None:
            model_id = self.get_default_tts_model_id()
        speech = self.client_openai.audio.speech.create(
            model=model_id,
            voice=voice,
            response_format=response_format,
            input=text,
        )
        from io import BytesIO

        bytes_io = BytesIO()
        for chunk in speech.iter_bytes():
            bytes_io.write(chunk)
        bytes_io.seek(0)
        return bytes_io.read()
