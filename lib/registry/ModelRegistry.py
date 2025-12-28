from typing import Any, TYPE_CHECKING
import os
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lib.providers.base_provider_config import BaseProviderConfig

class ModelRegistry:
    """
    This class provides a central in-memory structure to store and retrieve
    language model instances organized by provider (e.g., "openai", "anthropic").

    Models are not registered manually. Instead, each provider configuration
    (e.g., `OpenAIConfig`, `AnthropicConfig`) defines its available models and
    exposes a `register_models()` method. Calling this method will automatically
    populate the registry with all models for that provider.

    Features:
    - Retrieve specific models by provider and model name.
    - List all registered providers.
    - List all models available under a given provider.

    Notes:
        - Provider names are treated case-insensitively and stored in lowercase.
        - For details on how to implement a new provider, see the documentation
        of `BaseProviderConfig`.
        - Some providers and their models (e.g., OpenAI, Anthropic) are already
        implemented under `lib.providers` and can be used directly without
        additional setup. Remember to check for import of lib.providers in case of issues.
        - Some providers may expose only a subset of available models for efficiency.
        You can use `add_model()` and `remove_model()` methods to adjust the registry 
        as needed for your project.
    """
    _registry: dict[str, dict[str, Any]] = {}

    @classmethod
    def register_model(cls, provider: str, model_name: str, model: Any):
        """Register a model under the given provider name."""
        provider_key = provider.lower()
        cls._registry.setdefault(provider_key, {})[model_name] = model

    @classmethod
    def get_model(cls, provider: str, model_name: str) -> Any:
        """Retrieve a model instance by provider and model name."""
        return cls._registry.get(provider.lower(), {}).get(model_name)
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """Return a list of all registered providers."""
        return list(cls._registry.keys())

    @classmethod
    def list_models(cls, provider: str) -> list[str]:
        """Return a list of all models available for the given provider."""
        return list(cls._registry.get(provider.lower(), {}).keys())
    
    @classmethod
    def register_provider(cls, provider: "BaseProviderConfig"):
        """
        Register models from a provider config into the global registry,
        but only if the provider's API key is available and the provider
        is not already registered.
        """
        pname = provider.provider_name.lower()
        env_key = f"{pname.upper()}_API_KEY"

        if not os.environ.get(env_key):
            logger.warning("Skipping provider '%s': missing %s", pname, env_key)
            return

        if pname in cls._registry:
            raise ValueError(f"Provider '{pname}' is already registered.")

        provider.register_models()
        logger.info("Registered provider '%s'", pname)

    @classmethod
    def add_model(cls, provider_name: str, model_name: str, model_obj: object):
        """
        Register a new model under an existing provider.
        """
        provider_name = provider_name.lower()

        if provider_name not in cls._registry:
            raise ValueError(
                f"Provider '{provider_name}' is not registered. "
                f"Available providers: {list(cls._registry.keys())}"
            )

        if model_name in cls._registry[provider_name]:
            raise ValueError(
                f"Model '{model_name}' is already registered under provider '{provider_name}'."
            )
        cls._registry[provider_name][model_name] = model_obj
        logger.info("Model '%s' added to provider '%s'", model_name, provider_name)

    @classmethod
    def remove_model(cls, provider_name: str, model_name: str):
        provider_name = provider_name.lower()
        if  provider_name in cls._registry and model_name in cls._registry[provider_name]:
            del cls._registry[provider_name][model_name]
            logger.info("Model '%s' removed from provider '%s'", model_name, provider_name)
        else:
            logger.warning("Model '%s' not found under provider '%s'. Nothing to remove.", model_name, provider_name)