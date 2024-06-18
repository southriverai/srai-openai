from typing import List, Optional

from openai import OpenAI
from srai_core.tools_env import get_string_from_env


class ClientOpenaiEmbedding:
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = get_string_from_env("OPENAI_API_KEY")
        self.client_openai = OpenAI(api_key=api_key)

    def get_default_model_id(self) -> str:
        return "text-embedding-3-small"

    def list_model_id(self) -> list:
        return ["text-embedding-3-small", "text-embedding-3-large"]

    def get_embedding(
        self,
        text: str,
        *,
        model_id: Optional[str] = None,
    ) -> List[float]:
        if model_id is None:
            model_id = self.get_default_model_id()
        embedding = self.client_openai.embeddings.create(input=text, model=model_id)
        return embedding.data[0].embedding
