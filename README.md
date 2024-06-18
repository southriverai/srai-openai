# srai-openai
A set of functions to better interact with openai

wraps the promtps for text, images, whisper and function in a convinient way:


## Install
pip install srai-openai

## Use

### prompt text
'''
from srai_openai.client_openai_chatgpt import ClientOpenaiChatgpt
client = ClientOpenaiChatgpt()
print(client.prompt_default("You are a helpfull assitent", "This is a test"))
'''

### prompt image
'''
from srai_openai.client_openai_chatgpt import ClientOpenaiChatgpt
client = ClientOpenaiChatgpt()
system_message_prompt = "You are a helpfull assistent"
user_message_prompt = "What is shown here?"
with open("image.png", "rb") as file:
    image_base64 = base64.b64encode(file.read()).decode("utf-8")
print(client.prompt_default(system_message_prompt, user_message_prompt, image_base64=image_base64))
'''

### transcription
'''
client = ClientOpenaiWhisper()
transcription = client.transcription("test.mp3")
print(transcription["text"])
'''
