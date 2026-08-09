import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class BaseLLMClient(ABC):

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        raise NotImplementedError


class OpenAIClient(BaseLLMClient):

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("CAMD_MODEL", "gpt-5.5")

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Please add it to your .env file."
            )

        self.client = OpenAI(api_key=self.api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
        )

        return response.output_text