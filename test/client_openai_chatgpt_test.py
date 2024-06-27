import base64
import json
import os
from enum import Enum
from typing import List, Literal

from srai_openai.client_openai_chatgpt import ClientOpenaiChatgpt, PromptConfig
from srai_openai.model.chatgpt_tool import ChatgptTool, create_chatgpt_tool


def test_list_model_id():
    print("test_list_model_id")
    client = ClientOpenaiChatgpt()
    list_model_id = client.list_model_id()
    for model_id in list_model_id:
        print(model_id)


def test_prompt_default():
    print("test_prompt_default")
    system_message_content = "You are a helpfull assitent"
    user_message_content = "This is a test"
    client = ClientOpenaiChatgpt()
    print(client.prompt_default(system_message_content, user_message_content))


def test_prompt_default_json():
    client = ClientOpenaiChatgpt()
    result = client.prompt_default_json(
        "You are a helpfull assistent", "I'm fine, thank you. How are you? onlt respond in json"
    )
    print(json.dumps(result, indent=4))


def test_prompt_default_image():
    client = ClientOpenaiChatgpt()
    system_message_content = "You are a helpfull assistent"
    user_message_content = "What is shown here?"
    path_file_image = os.path.join("test", "data", "screen.png")
    with open(path_file_image, "rb") as file:
        image_base64 = base64.b64encode(file.read()).decode("utf-8")

    response = client.prompt_default(system_message_content, user_message_content, image_base64=image_base64)
    print(response)


def test_prompt_default_tool_enum():
    print("test_prompt_default_tool")
    system_message_content = "You are a helpfull assitent"
    user_message_content = "What is the weather like today in New York and Sofia?"

    class UnitEnum(Enum):
        celsius = "celsius"
        fahrenheit = "fahrenheit"

    def get_weather(location: str, unit: UnitEnum = UnitEnum.celsius) -> str:
        """Get the current weather in a given location.

        Args:
            location (str): Location to get the weather for.
            unit (UnitEnum, optional): The unit to return the temperature in. Default is celsius.
        Returns:
            str: weather for the location
        """
        return f"The weather in {location} is sunny today in {unit}."

    list_tool: List[ChatgptTool] = [create_chatgpt_tool(get_weather)]
    client = ClientOpenaiChatgpt()
    response = client.prompt_default_tool(system_message_content, user_message_content, list_tool)
    print(response)


def test_prompt_default_tool_literal():
    print("test_prompt_default_tool")
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
    client = ClientOpenaiChatgpt()
    response = client.prompt_default_tool(system_message_content, user_message_content, list_tool)
    print(response)


def test_prompt_config():
    print("test_prompt_config_default")
    system_message_content = "You are a helpfull assitent"
    user_message_content = "This is a test"
    client = ClientOpenaiChatgpt()
    prompt_config = PromptConfig.create(client.get_default_model_id(), system_message_content)
    prompt_config = prompt_config.append_user_message(user_message_content)
    print(client.prompt_for_prompt_config(prompt_config))


if __name__ == "__main__":
    # test_list_model_id()
    # test_prompt_default()
    # test_prompt_default_json()
    # test_prompt_default_image()
    test_prompt_default_tool_enum()
    # test_prompt_default_tool_literal()
    # test_prompt_config()
