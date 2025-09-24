from typing import Optional, Dict, Any
from pydantic_ai import Agent
from lib.config.ModelsConfig import ModelsConfig

cfg = ModelsConfig()

class PromptHandler:
    """Handle prompts and generate responses using configured LLMs."""

    def __init__(
        self,
        model_name: str,
        provider: Optional[str] = None,         
        system_prompt: Optional[str] = None,
    ):
        #provider -> {alias -> model} map
        provider_map: Dict[str, Dict[str, Any]] = {
            "openai": cfg.OpenAI,
            # "Anthropic": cfg.Anthropic,  
        }

        #provider key (case-insensitive)
        provider_key = provider.lower() if provider else "openai"

        if provider_key not in provider_map:
            raise ValueError(f"Unsupported provider '{provider}'. Available providers: {list(provider_map.keys())}")

        self.provider = provider_map[provider_key]

        #get selected model from the provider's config
        self.model = self.provider.get(model_name)
        if self.model is None:
            raise ValueError(f"Model '{model_name}' not found in provider '{provider_key}'. "
                             f"Available models: {list(self.provider.keys())}")

        #agent with optional system prompt
        self.agent = Agent(model=self.model, system_prompt=system_prompt)

    async def generate_response(self, prompt: str) -> str:
        """Async usage (e.g., Jupyter): await handler.generate_response(...)."""
        response = await self.agent.run(prompt)
        return response