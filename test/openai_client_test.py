import os
from srai_openai.client_openai import ClientOpenai
import openai

api_key = os.environ.get("OPENAI_API_KEY")
list_message = []
list_message.append({"role": "system", "content": "You are a helpfull assitent"})
list_message.append({"role": "user", "content": "This is a test"})


openai.api_key = api_key


client = ClientOpenai(api_key=api_key, call_cache=None)
call_dict = {"model": "gpt-3.5-turbo-0613", "messages": list_message, "max_tokens": 5}
print(client.call(call_dict))

list_engine_name = client.list_engine_name_gpt()
for engine_name in list_engine_name:
    print(engine_name)


client.call_list_message(list_message=list_message)
