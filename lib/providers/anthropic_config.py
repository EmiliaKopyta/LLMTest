from lib.config.ProviderKeys import ProviderKeys
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from lib.providers.base_provider_config import BaseProviderConfig

class AnthropicConfig(BaseProviderConfig):
    def __init__(self):
        keys = ProviderKeys()
        api_key = self.require_key(keys.keys.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY")
        super().__init__(api_key)
        self._provider = AnthropicProvider(api_key=api_key)

    @property
    def provider_name(self):
        return "anthropic"

    def get_models(self):
        return {
            "claude-3-haiku": AnthropicModel("claude-3-haiku-20240307", provider=self._provider),
            "claude-3-5-haiku": AnthropicModel("claude-3-5-haiku-20241022", provider=self._provider),
            "claude-sonnet-4": AnthropicModel("claude-sonnet-4-20250514", provider=self._provider),
            "claude-opus-4": AnthropicModel("claude-opus-4-20250514", provider=self._provider),
            "claude-opus-4-1": AnthropicModel("claude-opus-4-1-20250805", provider=self._provider),
            "claude-sonnet-4-5": AnthropicModel("claude-sonnet-4-5-20250929", provider=self._provider),
            "claude-haiku-4-5": AnthropicModel("claude-haiku-4-5-20251001", provider=self._provider),
            "claude-opus-4-5": AnthropicModel("claude-opus-4-5-20251101", provider=self._provider),
        }