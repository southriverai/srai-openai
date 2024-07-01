import json
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from srai_core.tools_env import get_string_from_env

from srai_openai.model.chatgpt_tool import ChatgptTool
from srai_openai.model.prompt_config import ChatgptEvent, PromptConfig


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
    ) -> str:
        if model is None:
            model = self.get_default_model_id()

        prompt_config_input = PromptConfig.create(model, system_message_content)
        prompt_config_input = prompt_config_input.append_user_message(user_message_content, image_base64)
        return self.prompt_for_prompt_config(prompt_config_input).last_message_text

    def prompt_default_json(
        self,
        system_message_content: str,
        user_message_content: str,
        *,
        model: Optional[str] = None,
        image_base64: Optional[str] = None,
    ) -> dict:

        if model is None:
            model = self.get_default_model_id()

        prompt_config_input = PromptConfig.create(model, system_message_content)
        prompt_config_input = prompt_config_input.append_user_message(
            user_message_content, image_base64, response_format={"type": "json_object"}
        )
        return json.loads(self.prompt_for_prompt_config(prompt_config_input).last_message_text)

    def complete_tool_call_requests(
        self, dict_chatgpt_tool: Dict[str, ChatgptTool], prompt_config_input: PromptConfig
    ) -> PromptConfig:
        if prompt_config_input.list_event[-1].event_type != ChatgptEvent.TOOL_CALL_REQUEST:
            raise Exception("last event is not a tool call request")

        list_tool_call_result = []
        for tool_call_request in prompt_config_input.list_event[-1].event_message["tool_calls"]:
            id = tool_call_request["id"]
            name = tool_call_request["function"]["name"]
            arguments = json.loads(tool_call_request["function"]["arguments"])

            chatgpt_tool = dict_chatgpt_tool[name]

            tool_call_result = {
                "id": id,
                "name": name,
                "result": chatgpt_tool.call(**arguments),
            }
            list_tool_call_result.append(tool_call_result)
        return prompt_config_input.append_tool_call_result(list_tool_call_result)

    def prompt_default_tool(
        self,
        system_message_content: str,
        user_message_content: str,
        list_chatgpt_tool: List[ChatgptTool],
        *,
        image_base64: Optional[str] = None,
        tool_choice: Optional[str] = "auto",
        model: Optional[str] = None,
    ) -> str:
        if model is None:
            model = self.get_default_model_id()

        dict_chatgpt_tool = {}
        list_tool_offer = []
        for chatgpt_tool in list_chatgpt_tool:
            dict_chatgpt_tool[chatgpt_tool.name] = chatgpt_tool
            list_tool_offer.append(chatgpt_tool.to_tool_dict())
        prompt_config_input = PromptConfig.create(model, system_message_content)
        prompt_config_input = prompt_config_input.append_user_message(
            user_message_content,
            image_base64=image_base64,
            list_tool_offer=list_tool_offer,
            tool_choice=tool_choice,
        )

        prompt_config_result = self.prompt_for_prompt_config(prompt_config_input)
        if prompt_config_result.list_event[-1].event_type != ChatgptEvent.TOOL_CALL_REQUEST:
            return prompt_config_result.last_message_text

        prompt_config_input_tools = self.complete_tool_call_requests(dict_chatgpt_tool, prompt_config_result)
        prompt_config_result_tools = self.prompt_for_prompt_config(prompt_config_input_tools)
        return prompt_config_result_tools.last_message_text

    def prompt_for_prompt_config(self, prompt_config_input: PromptConfig) -> PromptConfig:
        completion = self.client_openai.chat.completions.create(
            model=prompt_config_input.model_id,
            messages=prompt_config_input.messages,  # type: ignore
            tools=prompt_config_input.tools,  # type: ignore
            tool_choice=prompt_config_input.tool_choice,  # type: ignore
            response_format=prompt_config_input.response_format,  # type: ignore
        )
        if completion.choices[0].finish_reason == "tool_calls":
            return prompt_config_input.append_tool_call_request(
                completion.choices[0].message,
            )
        else:
            return prompt_config_input.append_assistent_message(
                completion.choices[0].message,
            )

    def options_for_prompt_config(
        self, prompt_config_input: PromptConfig, max_tokens: Optional[int] = 1
    ) -> List[Tuple[float, PromptConfig]]:
        completion = self.client_openai.chat.completions.create(
            model=prompt_config_input.model_id,
            messages=prompt_config_input.messages,  # type: ignore
            logprobs=True,
            top_logprobs=20,
            max_tokens=1,
        )
        model_dump = completion.choices[0].model_dump()
        list_option = []
        for option in model_dump["logprobs"]["content"][0]["top_logprobs"]:
            prompt_config_result = prompt_config_input.append_assistent_message(option["token"])
            list_option.append((option["logprob"], prompt_config_result))
        return list_option

    def prompt_default_image(self, prompt_config_input: PromptConfig) -> str:
        prompt_config_result = self.prompt_for_prompt_config(prompt_config_input)
        return prompt_config_result.last_message_text
