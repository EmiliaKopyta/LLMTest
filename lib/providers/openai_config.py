from lib.registry.ModelRegistry import ModelRegistry
from lib.config.ProviderKeys import ProviderKeys
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from lib.providers.base_provider_config import BaseProviderConfig

class OpenAIConfig(BaseProviderConfig):
    def __init__(self):
        self._provider = OpenAIProvider(api_key=ProviderKeys.OPENAI_API_KEY)

    @property
    def provider_name(self):
        return "openai"

    def get_models(self):
        return {
            "gpt-4o": OpenAIChatModel(provider=self._provider, model_name="gpt-4o"),
            "gpt-4o-mini": OpenAIChatModel(provider=self._provider, model_name="gpt-4o-mini")
        }

    def register_models(self):
        for name, model in self.get_models().items():
            ModelRegistry.register_model(self.provider_name, name, model)
