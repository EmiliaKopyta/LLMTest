from .ModelRegistry import ModelRegistry
from lib.providers import OpenAIConfig, AnthropicConfig, GoogleConfig, OpenRouterConfig
import logging

logger = logging.getLogger(__name__)

def register_all_models():
    """
    Register all available provider configurations in the global ModelRegistry.

    Iterates over the built‑in provider classes (OpenAI, Anthropic, Google, OpenRouter),
    instantiates each configuration, and attempts to register it.
    """
    for ProviderCls in [OpenAIConfig, AnthropicConfig, GoogleConfig, OpenRouterConfig]:
        try:
            config = ProviderCls()
            ModelRegistry.register_provider(config)
        except ValueError as e:
            logger.warning("Skipping provider '%s': %s", ProviderCls.__name__ , e)
