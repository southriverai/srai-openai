import json
from copy import copy
from typing import List, Optional

import tiktoken
from openai.types.chat.chat_completion_message import ChatCompletionMessage


class ChatgptEvent:
    SYSTEM_MESSAGE = "system_message"
    ASSISTENT_MESSAGE = "assistent_message"
    USER_MESSAGE = "user_message"
    TOOL_CALL_REQUEST = "tool_call_request"
    TOOL_CALL_RESULT = "tool_call_result"
    list_event_type = [SYSTEM_MESSAGE, ASSISTENT_MESSAGE, USER_MESSAGE, TOOL_CALL_REQUEST, TOOL_CALL_RESULT]

    def __init__(
        self,
        event_type: str,
        event_message: dict,
        *,
        list_tool_offer: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[dict] = None,
    ) -> None:
        if event_type not in ChatgptEvent.list_event_type:
            raise Exception(f"type {type} not in {str(ChatgptEvent.list_event_type)}")
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
        self.response_format = response_format

    def is_text_message(self) -> bool:
        return self.event_type in [
            ChatgptEvent.SYSTEM_MESSAGE,
            ChatgptEvent.ASSISTENT_MESSAGE,
            ChatgptEvent.USER_MESSAGE,
        ]

    def __str__(self) -> str:
        if (
            self.event_type == ChatgptEvent.SYSTEM_MESSAGE
            or self.event_type == ChatgptEvent.ASSISTENT_MESSAGE
            or self.event_type == ChatgptEvent.USER_MESSAGE
        ):
            content = self.event_message["content"][0]["text"]  # type: ignore
            if self.event_type == ChatgptEvent.SYSTEM_MESSAGE:
                return f"System message:\n    {content}"
            elif self.event_type == ChatgptEvent.ASSISTENT_MESSAGE:
                return f"Assistent message:\n    {content}"
            else:
                return f"User message:\n    {content}"
        elif self.event_type == ChatgptEvent.TOOL_CALL_REQUEST:
            return f"Tool call request:\n    {json.dumps(self.event_message, indent=4)}"
        elif self.event_type == ChatgptEvent.TOOL_CALL_RESULT:
            return f"Tool call result:\n    {json.dumps(self.event_message, indent=4)}"
        else:
            return f"Event Type: {self.event_type}, Event Content: {self.event_message}"

    def to_dict(self) -> dict:
        dict_event = {
            "type": self.event_type,
            "message": self.event_message,
        }
        if self.list_tool_offer is not None:
            dict_event["list_tool_offer"] = self.list_tool_offer
        if self.tool_choice is not None:
            dict_event["tool_choice"] = self.tool_choice
        if self.response_format is not None:
            dict_event["response_format"] = self.response_format
        return dict_event

    @staticmethod
    def from_dict(event_dict: dict) -> "ChatgptEvent":
        event_type = event_dict["type"]
        event_message = event_dict["message"]
        list_tool_offer = event_dict.get("list_tool_offer")
        tool_choice = event_dict.get("tool_choice")
        response_format = event_dict.get("response_format")
        return ChatgptEvent(
            event_type,
            event_message,
            list_tool_offer=list_tool_offer,
            tool_choice=tool_choice,
            response_format=response_format,
        )


class PromptConfig:
    def __init__(
        self,
        model_id: str,
        list_event: List[ChatgptEvent],
        tool_choice: Optional[str] = None,
    ) -> None:

        if model_id is None:
            raise Exception("model is None")
        if list_event is None:
            raise Exception("list_event is None")
        self.model_id = model_id
        self.list_event = copy(list_event)

    @property
    def last_event(self) -> ChatgptEvent:
        return self.list_event[-1]

    @property
    def tools(self) -> Optional[List[dict]]:
        return self.last_event.list_tool_offer

    @property
    def tool_choice(self) -> Optional[str]:
        return self.last_event.tool_choice

    @property
    def response_format(self) -> Optional[dict]:
        return self.last_event.response_format

    @property
    def tool_calls(self) -> List[dict]:
        if self.last_event.event_type != ChatgptEvent.TOOL_CALL_REQUEST:
            raise Exception("last event is not a tool call request")
        if self.last_event.event_message is None:
            raise Exception("last event message is None")
        return self.last_event.event_message["tool_calls"]

    @property
    def last_message_content(self) -> str:
        return self.messages[-1]["content"]

    @property
    def last_message_text(self) -> str:
        return self.list_text_message[-1]["content"][0]["text"]

    @property
    def list_text_message(self) -> List[dict]:
        list_message = []
        for event in self.list_event:
            if event.is_text_message():
                list_message.append(event.event_message)
        return list_message

    @property
    def messages(self) -> List[dict]:
        return [event.event_message for event in self.list_event]  # type: ignore

    @staticmethod
    def create_message_text(role: str, message_text: str) -> dict:
        message_content = [
            {
                "type": "text",
                "text": message_text,
            },
        ]
        return PromptConfig.create_message(role, message_content)

    @staticmethod
    def create_message(role: str, message_content: List[dict]) -> dict:
        if role not in ["system", "user", "assistant"]:
            raise Exception(f"role {role} not in [system, user, assistant]")
        return {"role": role, "content": message_content}

    def append_system_message(self, message_content: str) -> "PromptConfig":
        event_message = PromptConfig.create_message_text("system", message_content)
        event = ChatgptEvent(ChatgptEvent.SYSTEM_MESSAGE, event_message=event_message)
        prompt_config = PromptConfig(self.model_id, self.list_event)
        prompt_config.list_event.append(event)
        return prompt_config

    def append_user_message(
        self,
        message_text: str,
        image_base64: Optional[str] = None,
        response_format: Optional[dict] = None,
        list_tool_offer: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
    ) -> "PromptConfig":
        event_message = PromptConfig.create_message_text("user", message_text)
        if image_base64 is not None:
            if self.model_id != "gpt-4o":
                raise Exception(f"model_id: {self.model_id} not supported for image")
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
        prompt_config = PromptConfig(self.model_id, self.list_event)
        prompt_config.list_event.append(event)
        return prompt_config

    def append_assistent_message(
        self, message: ChatCompletionMessage, list_tool_call_result: List[dict] = []
    ) -> "PromptConfig":
        if message.role != "assistant":
            raise Exception("message role is not assistant")
        message_content = message.model_dump()["content"]
        if type(message_content) is str:
            event_message = PromptConfig.create_message_text("assistant", message_content)
        else:
            event_message = PromptConfig.create_message("assistant", message_content)
        event = ChatgptEvent(ChatgptEvent.ASSISTENT_MESSAGE, event_message=event_message)
        prompt_config = PromptConfig(self.model_id, self.list_event)
        prompt_config.list_event.append(event)
        return prompt_config

    def append_assistent_token(self, token: str) -> "PromptConfig":
        if self.list_event[-1].event_type != ChatgptEvent.ASSISTENT_MESSAGE:
            raise Exception("last event is not an assistent message")
        event_message_text = self.list_event[-1].event_message["content"][0]["text"]
        event_message = PromptConfig.create_message("assistant", event_message_text + token)
        event = ChatgptEvent(ChatgptEvent.ASSISTENT_MESSAGE, event_message=event_message)
        prompt_config = PromptConfig(self.model_id, self.list_event[:-1])
        prompt_config.list_event.append(event)
        return prompt_config

    def append_tool_call_request(
        self,
        message: ChatCompletionMessage,
    ) -> "PromptConfig":
        event = ChatgptEvent(
            ChatgptEvent.TOOL_CALL_REQUEST,
            event_message=message.model_dump(),
        )
        prompt_config = PromptConfig(self.model_id, self.list_event)
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
        prompt_config = PromptConfig(self.model_id, self.list_event)
        prompt_config.list_event.extend(list_event)
        return prompt_config

    def token_count(self) -> int:
        content = ""
        for message in self.messages:
            content += message["content"]
        if self.model_id == "gpt-4o":
            encoding_name = "cl100k_base"
        elif self.model_id == "gpt-4":
            encoding_name = "cl100k_base"
        elif self.model_id == "gpt-4-turbo-preview":
            encoding_name = "cl100k_base"
        elif self.model_id == "gpt-3.5-turbo":
            encoding_name = "cl100k_base"
        else:
            raise Exception(f"model {self.model_id} not supported")
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(content))
        return num_tokens

    def token_count_max(self) -> int:
        if self.model_id == "gpt-4":
            return 128000
        elif self.model_id == "gpt-4-turbo-preview":
            return 128000
        elif self.model_id == "gpt-3.5-turbo":
            return 16385
        else:
            raise Exception(f"model {self.model_id} not supported")

    @staticmethod
    def create(
        model_id: str,
        system_message_text: str,
    ) -> "PromptConfig":
        event_message = PromptConfig.create_message_text("system", system_message_text)
        event = ChatgptEvent("system_message", event_message=event_message)

        return PromptConfig(model_id, [event])

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "list_event": [event.to_dict() for event in self.list_event],
        }

    @staticmethod
    def from_dict(prompt_config_dict: dict) -> "PromptConfig":
        model_id = prompt_config_dict["model_id"]
        list_event = [ChatgptEvent.from_dict(event_dict) for event_dict in prompt_config_dict["list_event"]]
        return PromptConfig(model_id, list_event)

    def __str__(self) -> str:
        string_description = ""
        for event in self.list_event:
            string_description += str(event) + "\n"
        return string_description
