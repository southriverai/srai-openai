from typing import List, Optional
import openai
from srai_core.jsondict_store import JsondictStore


class ClientOpenai:
    def __init__(self, api_key: str, call_cache: Optional[JsondictStore] = None):
        self.api_key = api_key
        self.call_cache = call_cache

    def call(self, call_dict: dict, ignore_cache: bool = False):
        openai.api_key = self.api_key
        if not ignore_cache and self.call_cache is not None:
            if self.call_cache.exists(call_dict):
                return self.call_cache.read_jsondict(call_dict)

        call_dict_key = call_dict.copy()
        call_dict_key["api_key"] = self.api_key
        response = openai.ChatCompletion.create(**call_dict_key)
        if not ignore_cache and self.call_cache is not None:
            self.call_cache.write_jsondict(call_dict, response)
        return response

    def call_list_message(
        self,
        list_message: List[dict],
        engine_name: str = "gpt-3.5-turbo-0613",
        ignore_cache: bool = False,
    ):
        call_dict = {"model": engine_name, "messages": list_message}
        return self.call(call_dict, ignore_cache)

    # likely these 4 are all the same
    def list_engine_name(self):
        response = openai.Engine.list(api_key=self.api_key)
        return [engine["id"] for engine in response["data"]]

    def list_engine_name_gpt(self):
        response = openai.Engine.list(api_key=self.api_key)
        return [engine["id"] for engine in response["data"] if "gpt" in engine["id"]]

    def list_model_name(self):
        response = openai.Model.list(api_key=self.api_key)
        return [engine["id"] for engine in response["data"]]

    def list_model_name_gpt(self):
        response = openai.Model.list(api_key=self.api_key)
        return [engine["id"] for engine in response["data"] if "gpt" in engine["id"]]
