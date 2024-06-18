import base64
import json
import os

from srai_openai.client_openai_whisper import ClientOpenaiWhisper


def test_transcription():
    print("test_transcription")
    client = ClientOpenaiWhisper()
    transcription = client.transcription("test.mp3")
    print(transcription["text"])


if __name__ == "__main__":
    test_transcription()
