import os

from srai_openai.client_openai_audio import ClientOpenaiAudio


def test_text_to_speech():
    print("test_text_to_speech")
    client = ClientOpenaiAudio()
    client.text_to_speech_for_file("Hello, my name is John", os.path.join("test", "data", "test.mp3"))


def test_transcription():
    print("test_transcription")
    client = ClientOpenaiAudio()
    transcription = client.transcription(os.path.join("test", "data", "test.mp3"))
    print(transcription["text"])


if __name__ == "__main__":
    test_transcription()
    # test_text_to_speech()
