# srai-openai
A set of functions to better interact with openai

wraps the promtps for text, images, whisper and function in a convinient way:


## Install
pip install srai-openai

## Use

### prompt text
```python
from srai_openai.client_openai_chatgpt import ClientOpenaiChatgpt
client = ClientOpenaiChatgpt()
print(client.prompt_default("You are a helpfull assitent", "This is a test"))
```


### prompt json response
```python
from srai_openai.client_openai_chatgpt import ClientOpenaiChatgpt
import json
client = ClientOpenaiChatgpt()
result = client.prompt_default_json(
    "You are a helpfull assistent", "I'm fine, thank you. How are you? Respond in valid json"
)
print(json.dumps(result, indent=4))
```

### prompt image
```python
from srai_openai.client_openai_chatgpt import ClientOpenaiChatgpt
import base64
client = ClientOpenaiChatgpt()
system_message_prompt = "You are a helpfull assistent"
user_message_prompt = "What is shown here?"
with open("image.png", "rb") as file:
    image_base64 = base64.b64encode(file.read()).decode("utf-8")
print(client.prompt_default(system_message_prompt, user_message_prompt, image_base64=image_base64))
```

### prompt with tool
```python
from srai_openai.client_openai_chatgpt import ClientOpenaiChatgpt
from srai_openai.model.chatgpt_tool import ChatgptTool, create_chatgpt_tool
from typing import List, Literal

client = ClientOpenaiChatgpt()
system_message_content = "You are a helpfull assitent"
user_message_content = "What is the weather like today in New York?"

def get_weather(location: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get the current weather in a given location.

    Args:
        location (str): Location to get the weather for.
        unit (Literal["celsius", "fahrenheit"], optional): The unit to return the temperature in. Default is celsius.
    Returns:
        str: weather for the location
    """
    return f"The weather in {location} is sunny today in {unit}."

list_tool: List[ChatgptTool] = [create_chatgpt_tool(get_weather)]
response = client.prompt_default_tool(system_message_content, user_message_content, list_tool)
print(response)
```

### transcription
```python
from srai_openai.client_openai_whisper import ClientOpenaiWhisper
client = ClientOpenaiWhisper()
transcription = client.transcription("test.mp3")
print(transcription["text"])
```
