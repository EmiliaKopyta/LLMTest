from abc import ABC, abstractmethod
from lib.registry.ModelRegistry import ModelRegistry
from typing import Any

class BaseProviderConfig(ABC):
    """
    Abstract base class for provider configuration.

    Subclasses must define:
    - `provider_name`: a property returning the provider's name (e.g. "openai", "anthropic").
    - `get_models()`: a method returning a dictionary of available models for this provider.

    In your subclass constructor, initialize the provider with API key.

    Notes:
        - `register_models()` is already implemented here and will automatically
        register all models returned by `get_models()` into the global ModelRegistry.
        -  While providers expose `register_models()`, it is recommended to register
        a new provider through `ModelRegistry.register_provider()` to avoid errors
        like duplicate provider names or missing API keys.
        `ModelRegistry.add_model()` and `ModelRegistry.remove_model()`
        - Adding or removing models to a provider globally should be done via
        `ModelRegistry.add_model()` and `ModelRegistry.remove_model()`.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self._models: dict[str, Any] = {}

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    def get_models(self) -> dict[str, Any]:
        pass

    def require_key(self, key: str | None, key_name: str) -> str:
        """Ensure that an API key is present. Raise a clear error if missing."""
        if not key:
            raise ValueError(
                f"Missing API key: {key_name}. "
                f"Please set the environment variable {key_name} before using {self.provider_name}."
            )
        return key

    def register_models(self):
        """Register models in ModelRegistry. Mainly used while adding a new provider."""
        for name, model in self.get_models().items():
            ModelRegistry.register_model(self.provider_name, name, model)
