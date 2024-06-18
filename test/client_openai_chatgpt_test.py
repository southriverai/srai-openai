from srai_openai.client_openai_chatgpt import ClientOpenaiChatgpt, PromptConfig


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


def test_prompt_config_default():
    print("test_prompt_config_default")
    system_message_content = "You are a helpfull assitent"
    user_message_content = "This is a test"
    client = ClientOpenaiChatgpt()
    prompt_config = PromptConfig.create(client.get_default_model_id(), system_message_content)
    prompt_config = prompt_config.append_user_message(user_message_content)
    print(client.prompt_for_prompt_config(prompt_config))


def test_prompt_json():
    client = ClientOpenaiChatgpt()
    result = client.prompt_json(
        "You are a helpfull assistent", "I'm fine, thank you. How are you? onlt respond in json"
    )
    print(json.dumps(result, indent=4))


def test_prompt_json_1():
    client = ClientOpenaiChatgpt()

    form_filled_dict = client.prompt_json(
        "You are a helpfull assistent",
        user_message_prompt,
        model="gpt-4o",
    )
    print(form_filled_dict)


if __name__ == "__main__":
    test_list_model_id()
    test_prompt_default()
