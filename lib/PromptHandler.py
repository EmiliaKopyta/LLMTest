from typing import Optional, Any
from pydantic_ai import Agent
from lib.registry.ModelRegistry import ModelRegistry

class PromptHandler:
    """
    Handle prompts and generate responses using configured LLMs.
    Looks up models from the global ModelRegistry based on provider and model name.
    """

    def __init__(
        self,
        model_name: str,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.provider_key = self._normalize_provider(provider)
        self.model = self._get_model(self.provider_key, model_name)
        self.agent = self._create_agent(self.model, system_prompt)

    def _normalize_provider(self, provider: Optional[str]) -> str:
        return provider.lower() if provider else "openai"

    def _validate_provider(self, provider_key: str):
        if provider_key not in ModelRegistry.list_providers():
            raise ValueError(
                f"Unsupported provider '{provider_key}'. "
                f"Available providers: {ModelRegistry.list_providers()}"
            )

    def _get_model(self, provider_key: str, model_name: str) -> Any:
        self._validate_provider(provider_key)
        model = ModelRegistry.get_model(provider_key, model_name)
        if model is None:
            raise ValueError(
                f"Model '{model_name}' not found in provider '{provider_key}'. "
                f"Available models: {ModelRegistry.list_models(provider_key)}"
            )
        return model

    def _create_agent(self, model: Any, system_prompt: Optional[str]) -> Agent:
        return Agent(model=model, system_prompt=system_prompt)

    async def generate_response(self, prompt: str) -> str:
        """Async usage (e.g., Jupyter): await handler.generate_response(...)."""
        response = await self.agent.run(prompt)
        return getattr(response, "data", str(response))


# from typing import Optional, Dict, Any
# from pydantic_ai import Agent
# from lib.registry.ModelRegistry import ModelRegistry

# class PromptHandler:
#     """Handle prompts and generate responses using configured LLMs."""

#     def __init__(
#         self,
#         model_name: str,
#         provider: Optional[str] = None,
#         system_prompt: Optional[str] = None,
#     ):
#         # Provider key (case-insensitive)
#         provider_key = provider.lower() if provider else "openai"

#         if provider_key not in ModelRegistry.list_providers():
#             raise ValueError(
#                 f"Unsupported provider '{provider}'. Available providers: {ModelRegistry.list_providers()}"
#             )

#         # Get selected model from the registry
#         self.model = ModelRegistry.get_model(provider_key, model_name)
#         if self.model is None:
#             raise ValueError(
#                 f"Model '{model_name}' not found in provider '{provider_key}'. "
#                 f"Available models: {ModelRegistry.list_models(provider_key)}"
#             )

#         self.agent = Agent(model=self.model, system_prompt=system_prompt)

#     async def generate_response(self, prompt: str) -> str:
#         """Async usage (e.g., Jupyter): await handler.generate_response(...)."""
#         response = await self.agent.run(prompt)
#         return getattr(response, "data", str(response))