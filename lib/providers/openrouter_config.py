from lib.config.ProviderKeys import ProviderKeys
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from lib.providers.base_provider_config import BaseProviderConfig

class OpenRouterConfig(BaseProviderConfig):
    def __init__(self):
        keys = ProviderKeys()
        api_key = self.require_key(keys.keys.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY")
        super().__init__(api_key)
        self._provider = OpenRouterProvider(api_key=api_key)

    @property
    def provider_name(self):
        return "openrouter"

    def get_models(self):
        return {
            "google/gemini-2.5-flash-lite": OpenRouterModel("google/gemini-2.5-flash-lite", provider=self._provider),
            "google/gemini-2.5-pro": OpenRouterModel("google/gemini-2.5-pro", provider=self._provider),
            "google/gemini-3-pro-preview": OpenRouterModel("google/gemini-3-pro-preview", provider=self._provider),
            "x-ai/grok-4-fast": OpenRouterModel("x-ai/grok-4-fast", provider=self._provider),
            "x-ai/grok-4.1-fast": OpenRouterModel("x-ai/grok-4.1-fast", provider=self._provider),
            "anthropic/claude-sonnet-4": OpenRouterModel("anthropic/claude-sonnet-4", provider=self._provider),
            "minimax/minimax-m2": OpenRouterModel("minimax/minimax-m2", provider=self._provider),
            "deepseek/deepseek-v3.2": OpenRouterModel("deepseek/deepseek-v3.2", provider=self._provider),
        }