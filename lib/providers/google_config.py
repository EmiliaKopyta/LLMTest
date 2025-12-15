from lib.config.ProviderKeys import ProviderKeys
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from lib.providers.base_provider_config import BaseProviderConfig

class GoogleConfig(BaseProviderConfig):
    def __init__(self):
        keys = ProviderKeys()
        api_key = self.require_key(keys.keys.get("GOOGLE_API_KEY"), "GOOGLE_API_KEY")
        super().__init__(api_key)
        self._provider = GoogleProvider(api_key=api_key)

    @property
    def provider_name(self):
        return "google"

    def get_models(self):
        return {
            "gemini-2.5-pro": GoogleModel(model_name="gemini-2.5-pro", provider=self._provider),
            "gemini-2.5-flash": GoogleModel(model_name="gemini-2.5-flash", provider=self._provider),
            "gemini-2.5-flash-lite": GoogleModel(model_name="gemini-2.5-flash-lite", provider=self._provider),
            "gemini-2.0-flash": GoogleModel(model_name="gemini-2.0-flash", provider=self._provider),
            "gemini-2.0-flash-lite": GoogleModel(model_name="gemini-2.0-flash-lite", provider=self._provider),
            "gemini-embedding-001": GoogleModel(model_name="gemini-embedding-001", provider=self._provider)
        }
