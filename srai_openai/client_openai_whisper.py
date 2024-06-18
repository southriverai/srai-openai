import json
from typing import Optional

from openai import OpenAI
from srai_core.tools_env import get_string_from_env


class ClientOpenaiWhisper:
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = get_string_from_env("OPENAI_API_KEY")
        self.client_openai = OpenAI(api_key=api_key)

    def get_default_model_id(self) -> str:
        return "whisper-1"

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
        return json.loads(transcription.model_dump_json())

    def transcription_by_wave(
        self,
        path_file_audio: str,
    ) -> dict:
        # TODO add script
        model_id = self.get_default_model_id()
        with open(path_file_audio, "rb") as file:
            transcription = self.client_openai.audio.transcriptions.create(
                model=model_id, file=file, response_format="verbose_json", timestamp_granularities=["word"]
            )
        return json.loads(transcription.model_dump_json())
