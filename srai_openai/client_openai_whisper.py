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
        return transcription.model_dump()

    def text_to_speech(
        self,
        text: str,
        *,
        model_id: Optional[str] = None,
    ) -> bytes:
        if model_id is None:
            model_id = self.get_default_model_id()
        speech = self.client_openai.audio.speech.create(
            model=model_id,
            voice="alloy",
            response_format="mp3",
            input=text,
        )
        return speech.write_to_file("temp.mp3")  # TODO fix this or stream output


# from pathlib import Path
# import openai

# speech_file_path = Path(__file__).parent / "speech.mp3"
# response = openai.audio.speech.create(
#     model="tts-1", voice="alloy", input="The quick brown fox jumped over the lazy dog."
# )
# response.stream_to_file(speech_file_path)
