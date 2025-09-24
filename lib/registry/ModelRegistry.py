from typing import Any


class ModelRegistry:
    from typing import Any

class ModelRegistry:
    """
    This class provides a central in-memory structure to store and retrieve
    language model instances organized by provider (e.g., "openai", "anthropic").

    All models must be registered using `register_model()` before they can be accessed.

    Features:
    - Register models under a provider name.
    - Retrieve specific models by provider and model name.
    - List all registered providers.
    - List all models available under a given provider.

    Example:
        ModelRegistry.register_model("openai", "gpt-4o", OpenAIChatModel(...))
        model = ModelRegistry.get_model("openai", "gpt-4o")

    Note:
        Provider names are treated case-insensitively and stored in lowercase.
    """
    _registry: dict[str, dict[str, Any]] = {}

    @classmethod
    def register_model(cls, provider: str, model_name: str, model: Any):
        provider_key = provider.lower()
        cls._registry.setdefault(provider_key, {})[model_name] = model

    @classmethod
    def get_model(cls, provider: str, model_name: str) -> Any:
        return cls._registry.get(provider.lower(), {}).get(model_name)

    @classmethod
    def list_providers(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def list_models(cls, provider: str) -> list[str]:
        return list(cls._registry.get(provider.lower(), {}).keys())
