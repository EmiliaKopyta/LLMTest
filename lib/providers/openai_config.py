from lib.config.ProviderKeys import ProviderKeys
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from lib.providers.base_provider_config import BaseProviderConfig

class OpenAIConfig(BaseProviderConfig):
    def __init__(self):
        keys = ProviderKeys()
        api_key = self.require_key(keys.keys.get("OPENAI_API_KEY"), "OPENAI_API_KEY")
        super().__init__(api_key)
        self._provider = OpenAIProvider(api_key=api_key)

    @property
    def provider_name(self):
        return "openai"

    def get_models(self):
        return {
            "gpt-5.1": OpenAIChatModel(model_name="gpt-5.1", provider=self._provider),
            "gpt-5-mini": OpenAIChatModel(model_name="gpt-5-mini", provider=self._provider),
            "gpt-4.1": OpenAIChatModel(model_name="gpt-4.1", provider=self._provider),
            "gpt-4o": OpenAIChatModel(model_name="gpt-4o", provider=self._provider),
            "gpt-4o-mini": OpenAIChatModel(model_name="gpt-4o-mini", provider=self._provider),
            "o1-pro": OpenAIChatModel(model_name="o1-pro", provider=self._provider),
            "o3": OpenAIChatModel(model_name="o3", provider=self._provider),
            "o3-mini": OpenAIChatModel(model_name="o3-mini", provider=self._provider),
        }
