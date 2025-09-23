from dataclasses import dataclass, field
from lib.config.ProviderKeys import ProviderKeys
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

@dataclass(frozen=True)
class ModelsConfig:
    """Class for keeping track of model configurations."""
    OpenAI: dict[str, OpenAIChatModel] = field(
        default_factory=lambda: {
            "gpt-4o": OpenAIChatModel(
                provider=OpenAIProvider(api_key=ProviderKeys.OPENAI_API_KEY),
                model_name="gpt-4o"
            ),
            "gpt-4o-mini": OpenAIChatModel(
                provider=OpenAIProvider(api_key=ProviderKeys.OPENAI_API_KEY),
                model_name="gpt-4o-mini"
            ),
        }
    )
    '''Anthropic: dict[str, AnthropicModel] = field(
        default_factory=lambda: {
            "claude-2": AnthropicModel(
                provider=AnthropicProvider(api_key=ProviderKeys.ANTHROPIC_API_KEY),
                model_name="claude-2"
            ),
            "claude-3": AnthropicModel(
                provider=AnthropicProvider(api_key=ProviderKeys.ANTHROPIC_API_KEY),
                model_name="claude-3"
            ),
        }
    )'''
