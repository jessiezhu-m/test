from typing import Optional, Dict, Any, Callable, Union
import os
from openai import AzureOpenAI, AsyncAzureOpenAI
from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, GeneratorOutput, CompletionUsage

class AzureOpenAIClient(ModelClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        chat_completion_parser: Optional[Callable] = None,
        env_api_key_name: str = "AZURE_OPENAI_API_KEY",
        env_endpoint_name: str = "AZURE_OPENAI_ENDPOINT",
        env_deployment_name: str = "AZURE_OPENAI_DEPLOYMENT",
    ):
        super().__init__()
        self.api_key = api_key or os.getenv(env_api_key_name)
        self.endpoint = endpoint or os.getenv(env_endpoint_name)
        self.deployment_name = deployment_name or os.getenv(env_deployment_name)
        self.api_version = api_version
        self.sync_client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
        )
        self.async_client = None
        self.chat_completion_parser = chat_completion_parser

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        if model_type == ModelType.LLM:
            api_kwargs = api_kwargs.copy()
            api_kwargs.setdefault("model", self.deployment_name)
            return self.sync_client.chat.completions.create(**api_kwargs)
        raise NotImplementedError("Only LLM supported in compact AzureOpenAIClient.")

    async def acall(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        if self.async_client is None:
            self.async_client = AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
            )
        if model_type == ModelType.LLM:
            api_kwargs = api_kwargs.copy()
            api_kwargs.setdefault("model", self.deployment_name)
            return await self.async_client.chat.completions.create(**api_kwargs)
        raise NotImplementedError("Only LLM supported in compact AzureOpenAIClient.")
