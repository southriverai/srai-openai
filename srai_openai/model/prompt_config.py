import json
from copy import copy
from typing import List, Optional

import tiktoken


class ChatgptEvent:
    SYSTEM_MESSAGE = "system_message"
    USER_MESSAGE = "user_message"
    TOOL_CALL_REQUEST = "tool_call_request"
    TOOL_CALL_RESULT = "tool_call_result"
    list_event_type = [SYSTEM_MESSAGE, USER_MESSAGE, TOOL_CALL_REQUEST, TOOL_CALL_RESULT]

    def __init__(
        self,
        event_type: str,
        *,
        event_message: Optional[dict] = {},
        list_tool_offer: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        list_tool_call_request: Optional[list[dict]] = None,
        response_format: Optional[dict] = None,
    ) -> None:
        if event_type not in ChatgptEvent.list_event_type:
            raise Exception(f"type {type} not in {str(ChatgptEvent.list_event_type)}")
        if event_type == ChatgptEvent.SYSTEM_MESSAGE or event_type == ChatgptEvent.USER_MESSAGE:
            if event_message == {}:
                raise Exception("Event_message is empty")
        if response_format is not None:
            if event_type != ChatgptEvent.USER_MESSAGE:
                raise Exception("can only format response for user messages")
            if response_format != {"type": "json_object"}:
                list_repsonse_format = [{"type": "json_object"}]
                raise Exception(f"response_format {response_format} is not allowed, must be in {list_repsonse_format}")
        if tool_choice is not None:
            if tool_choice not in ["none", "auto"]:
                raise Exception(f"tool_choice {tool_choice} not in [none, auto]")

        self.event_type = event_type
        self.event_message = event_message
        self.list_tool_offer = list_tool_offer
        self.tool_choice = tool_choice
        self.list_tool_call_request = list_tool_call_request
        self.response_format = response_format

    def is_text_message(self) -> bool:
        return self.event_type in [ChatgptEvent.SYSTEM_MESSAGE, ChatgptEvent.USER_MESSAGE]

    def __str__(self) -> str:
        if self.event_type == ChatgptEvent.SYSTEM_MESSAGE or self.event_type == ChatgptEvent.USER_MESSAGE:
            content = self.event_message["content"][0]["text"]  # type: ignore
            if self.event_type == ChatgptEvent.SYSTEM_MESSAGE:
                return f"System message:\n    {content}"
            else:
                return f"User message:\n    {content}"
        elif self.event_type == ChatgptEvent.TOOL_CALL_REQUEST:
            return f"Tool call request:\n    {json.dumps(self.list_tool_call_request, indent=4)}"
        elif self.event_type == ChatgptEvent.TOOL_CALL_RESULT:
            return f"Tool call result:\n    {json.dumps(self.event_message, indent=4)}"
        else:
            return f"Event Type: {self.event_type}, Event Content: {self.event_message}"


class PromptConfig:
    def __init__(
        self,
        model,
        list_event: List[ChatgptEvent],
        tool_choice: Optional[str] = None,
    ) -> None:

        if model is None:
            raise Exception("model is None")
        if list_event is None:
            raise Exception("list_event is None")
        if tool_choice is not None:
            if tool_choice not in ["none", "auto"]:
                raise Exception(f"tool_choice {tool_choice} not in [none, auto]")
        self.model = model
        self.list_event = copy(list_event)

    @property
    def tools(self) -> Optional[List[dict]]:
        return self.list_event[-1].list_tool_offer

    @property
    def last_message_text(self) -> str:
        return self.list_text_message[-1]["content"][0]["text"]  # TODO deal with messages without text

    @property
    def last_message_content(self) -> str:
        return self.messages[-1]["content"]

    @property
    def list_text_message(self) -> List[dict]:
        return [event.event_message for event in self.list_event if event.is_text_message()]

    @property
    def messages(self) -> List[dict]:
        return [event.event_message for event in self.list_event]  # type: ignore

    @staticmethod
    def create_message(role: str, message_content: str) -> dict:
        if role not in ["system", "user", "assistant"]:
            raise Exception(f"role {role} not in [system, user, assistant]")
        return {
            "role": role,
            "content": [
                {
                    "type": "text",
                    "text": message_content,
                },
            ],
        }

    def append_system_message(self, message_content: str) -> "PromptConfig":
        event_message = PromptConfig.create_message("system", message_content)
        event = ChatgptEvent(ChatgptEvent.SYSTEM_MESSAGE, event_message=event_message)
        prompt_config = PromptConfig(self.model, self.list_event)
        prompt_config.list_event.append(event)
        return prompt_config

    def append_user_message(
        self,
        message_content: str,
        image_base64: Optional[str] = None,
        response_format: Optional[dict] = None,
        list_tool_offer: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
    ) -> "PromptConfig":
        event_message = PromptConfig.create_message("user", message_content)
        if image_base64 is not None:
            if self.model != "gpt-4o":
                raise Exception(f"model {self.model} not supported for image")
            event_message["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )
        event = ChatgptEvent(
            ChatgptEvent.USER_MESSAGE,
            event_message=event_message,
            response_format=response_format,
            list_tool_offer=list_tool_offer,
            tool_choice=tool_choice,
        )
        prompt_config = PromptConfig(self.model, self.list_event)
        prompt_config.list_event.append(event)
        return prompt_config

    def append_assistent_message(self, message_content: str, list_tool_call_result: List[dict] = []) -> "PromptConfig":
        event_message = PromptConfig.create_message("assistant", message_content)
        event = ChatgptEvent(ChatgptEvent.SYSTEM_MESSAGE, event_message=event_message)
        prompt_config = PromptConfig(self.model, self.list_event)
        prompt_config.list_event.append(event)
        return prompt_config

    def append_tool_call_request(self, event_message: dict, list_tool_call_request: List[dict]) -> "PromptConfig":
        event = ChatgptEvent(
            ChatgptEvent.TOOL_CALL_REQUEST,
            event_message=event_message,
            list_tool_call_request=list_tool_call_request,
        )
        prompt_config = PromptConfig(self.model, self.list_event)
        prompt_config.list_event.append(event)
        return prompt_config

    def append_tool_call_result(self, list_tool_call_result: List[dict]) -> "PromptConfig":
        list_event: List[ChatgptEvent] = []
        for tool_call_result in list_tool_call_result:
            message = {
                "role": "tool",
                "tool_call_id": tool_call_result["id"],
                "name": tool_call_result["name"],
                "content": tool_call_result["result"],
            }
            list_event.append(ChatgptEvent(ChatgptEvent.TOOL_CALL_RESULT, event_message=message))
        prompt_config = PromptConfig(self.model, self.list_event)
        prompt_config.list_event.extend(list_event)
        return prompt_config

    def __str__(self) -> str:
        string_description = ""
        for event in self.list_event:
            string_description += str(event) + "\n"
        return string_description

    def token_count(self) -> int:
        content = ""
        for message in self.messages:
            content += message["content"]
        if self.model == "gpt-4o":
            encoding_name = "cl100k_base"
        elif self.model == "gpt-4":
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
        elif self.model == "gpt-4-turbo-preview":
            return 128000
        elif self.model == "gpt-3.5-turbo":
            return 16385
        else:
            raise Exception(f"model {self.model} not supported")

    @staticmethod
    def create(
        model: str,
        system_message_content: str,
    ) -> "PromptConfig":
        event_message = PromptConfig.create_message("system", system_message_content)
        event = ChatgptEvent("system_message", event_message=event_message)

        return PromptConfig(model, [event])
        return PromptConfig(model, [event])
