import json
from typing import Dict, List, Optional

import tiktoken
from openai import OpenAI
from openai.types.chat.completion_create_params import ResponseFormat
from srai_core.tools_env import get_string_from_env


class PromptConfig:
    def __init__(
        self,
        model: str,
        list_message: List[Dict[str, str]],
        list_list_tool_call_result: List[List[dict]] = [],
        list_tool: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[str] = None,
    ):
        if model is None:
            raise Exception("model is None")
        if list_message is None:
            raise Exception("list_message is None")
        if response_format is not None:
            if response_format not in ["json"]:
                raise Exception(f"response_format {response_format} not in [json]")
        if tool_choice is not None:
            if tool_choice not in ["none", "auto"]:
                raise Exception(f"tool_choice {tool_choice} not in [none, auto]")
        self.model = model
        self.list_message = list_message
        self.list_list_tool_call_result = list_list_tool_call_result
        self.list_tool = list_tool

        self.response_format = response_format
        self.tool_choice: Optional[str] = tool_choice

    def token_count(self) -> int:
        content = ""
        for message in self.list_message:
            content += message["content"]
        if self.model == "gpt-4":
            encoding_name = "cl100k_base"
        if self.model == "gpt-4":
            encoding_name = "cl100k_base"
        elif self.model == "gpt-4-turbo-preview":
            encoding_name = "cl100k_base"
        elif self.model == "gpt-3.5-turbo":
            encoding_name = "cl100k_base"
        else:
            raise Exception(f"model {self.model} not supported")
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(content))
        return num_tokens

    def token_count_max(self) -> int:
        if self.model == "gpt-4":
            return 128000
        if self.model == "gpt-4-turbo-preview":
            return 128000
        elif self.model == "gpt-3.5-turbo":
            return 16385
        else:
            raise Exception(f"model {self.model} not supported")

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "list_message": self.list_message,
            "list_list_tool_call_result": self.list_list_tool_call_result,
            "list_tool": self.list_tool,
            "response_format": self.response_format,
            "tool_choice": self.tool_choice,
        }

    def set_response_format(self, response_format: str) -> "PromptConfig":
        return PromptConfig(
            self.model,
            self.list_message,
            self.list_list_tool_call_result,
            self.list_tool,
            self.tool_choice,
            response_format,
        )

    def apppend_message(
        self,
        role: str,
        message_content: str,
        list_tool_call_result: List[dict] = [],
        image_base64: Optional[str] = None,
    ) -> "PromptConfig":
        if role not in ["system", "user", "assistant"]:
            raise Exception(f"role {role} not in [system, user, assistant]")
        list_message = self.list_message.copy()
        list_list_tool_call_result = self.list_list_tool_call_result.copy()

        if image_base64 is None:
            content = message_content
        else:
            if role != "user":
                raise Exception(f"role {role} not in [user]")
            image_base64 = image_base64
            content = [
                {"type": "text", "text": message_content},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ]
        list_message.append(
            {
                "role": role,
                "content": content,
            }
        )
        list_list_tool_call_result.append(list_tool_call_result)
        return PromptConfig(self.model, list_message, list_list_tool_call_result)

    def append_system_message(self, message_content: str) -> "PromptConfig":
        return self.apppend_message("system", message_content, [])

    def append_user_message(self, message_content: str, image_base64: Optional[str] = None) -> "PromptConfig":
        return self.apppend_message("user", message_content, [], image_base64)

    def append_assistent_message(self, message_content: str, list_tool_call_result: List[dict] = []) -> "PromptConfig":
        return self.apppend_message("assistant", message_content, list_tool_call_result)

    @staticmethod
    def create(
        model: str,
        system_message_content: str,
    ) -> "PromptConfig":
        list_message = []
        list_message.append({"role": "system", "content": system_message_content})

        return PromptConfig(model, list_message)

    @staticmethod
    def from_dict(dict_prompt_config: dict) -> "PromptConfig":
        model = dict_prompt_config["model"]
        list_message = dict_prompt_config["list_message"]
        list_list_tool_call_result = dict_prompt_config["list_list_tool_call_result"]
        list_tool = dict_prompt_config["list_tool"]
        response_format = dict_prompt_config["response_format"]
        tool_choice = dict_prompt_config["tool_choice"]
        return PromptConfig(
            model,
            list_message,
            list_list_tool_call_result,
            list_tool,
            tool_choice,
            response_format,
        )


class ClientOpenaiChatgpt:

    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = get_string_from_env("OPENAI_API_KEY")
        self.client_openai = OpenAI(api_key=api_key)

    def get_default_model_id(self) -> str:
        return "gpt-4o"

    def list_model_id(self) -> list:
        model_list = self.client_openai.models.list().data
        return [model.id for model in model_list]

    def prompt_default(
        self,
        system_message_content: str,
        user_message_content: str,
        *,
        model: Optional[str] = None,
        image_base64: Optional[str] = None,
        format_json: Optional[bool] = None,
    ) -> str:
        if model is None:
            model = self.get_default_model_id()

        prompt_config_input = PromptConfig.create(model, system_message_content)
        if image_base64 is not None:
            if model != "gpt-4o":
                raise Exception(f"model {model} not supported for image")

        prompt_config_input = prompt_config_input.append_user_message(user_message_content, image_base64)
        if format_json:
            prompt_config_input = prompt_config_input.set_response_format("json")
        prompt_config_result = self.prompt_for_prompt_config(prompt_config_input)
        return prompt_config_result.list_message[-1]["content"]

    def prompt_default_json(
        self,
        system_message_content: str,
        user_message_content: str,
        *,
        model: Optional[str] = None,
    ) -> dict:
        response_content = self.prompt_default(
            system_message_content, user_message_content, model=model, format_json=True
        )
        return json.loads(response_content)

    def prompt_default_tool(
        self,
        system_message_content: str,
        user_message_content: str,
        dict_tool: Dict[str, str],
        *,
        image_base64: Optional[str] = None,
        tool_choice: Optional[str] = "auto",
        model: Optional[str] = None,
        format_json: Optional[bool] = None,
    ) -> str:
        if model is None:
            model = self.get_default_model_id()

        prompt_config_input = PromptConfig.create(model, system_message_content)
        if image_base64 is not None:
            if model != "gpt-4o":
                raise Exception(f"model {model} not supported for image")

        prompt_config_input = prompt_config_input.append_user_message(user_message_content, image_base64)
        if format_json:
            prompt_config_input = prompt_config_input.set_response_format("json")
        list_tool = []
        for key, value in dict_tool.items():
            list_tool.append({"name": key, "value": value})

        prompt_config_input.list_tool = list_tool
        prompt_config_input.tool_choice = tool_choice
        prompt_config_result = self.prompt_for_prompt_config(prompt_config_input)

        return prompt_config_result.list_message[-1]["content"]

    def prompt_for_prompt_config(self, prompt_config_input: PromptConfig) -> PromptConfig:
        if prompt_config_input.response_format == "json":
            response_format: ResponseFormat = {"type": "json_object"}
            completion = self.client_openai.chat.completions.create(
                model=prompt_config_input.model,
                messages=prompt_config_input.list_message,  # type: ignore
                tools=prompt_config_input.list_tool,  # type: ignore
                tool_choice=prompt_config_input.tool_choice,  # type: ignore
                response_format=response_format,  # type: ignore
            )
        else:
            completion = self.client_openai.chat.completions.create(
                model=prompt_config_input.model,
                messages=prompt_config_input.list_message,  # type: ignore
                tools=prompt_config_input.list_tool,  # type: ignore
                tool_choice=prompt_config_input.tool_choice,  # type: ignore
            )
        print(completion.model_dump_json())
        return prompt_config_input.append_assistent_message(
            completion.choices[0].message.content,  # type: ignore
        )

    def prompt_default_image(self, prompt_config_input: PromptConfig) -> str:
        prompt_config_result = self.prompt_for_prompt_config(prompt_config_input)
        return prompt_config_result.list_message[-1]["content"]
