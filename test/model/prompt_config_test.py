from srai_openai.model.prompt_config import PromptConfig


def test_prompt_config():
    prompt_config = PromptConfig.create("gpt-4o", "You are a helpfull assistent")
    prompt_config_dict = prompt_config.to_dict()
    prompt_config = PromptConfig.from_dict(prompt_config_dict)
    assert prompt_config.model_id == "gpt-4o"
    assert prompt_config.last_message_text == "You are a helpfull assistent"
    prompt_config = prompt_config.append_user_message("Hello")
    assert prompt_config.model_id == "gpt-4o"
    assert prompt_config.last_message_text == "Hello"
    prompt_config_dict = prompt_config.to_dict()
    prompt_config = PromptConfig.from_dict(prompt_config_dict)
    assert prompt_config.model_id == "gpt-4o"
    assert prompt_config.last_message_text == "Hello"


if __name__ == "__main__":
    test_prompt_config()
