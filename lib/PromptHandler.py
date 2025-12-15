from typing import Optional, Any
from pydantic_ai import Agent, UserContent
from typing import Union, Sequence
from lib.registry.ModelRegistry import ModelRegistry

class PromptHandler:
    """
    Handle prompts and generate responses using configured LLMs.
    This class looks up models from the global ModelRegistry based on provider and model name,
    and wraps them in a `pydantic_ai.Agent`
    for prompt execution.

    Parameters:
        model_name (str): Name of the model to use (must be registered in ModelRegistry).
        provider (str, optional): Provider name (defaults to "openai").
        system_prompt (str, optional): Optional system-level instruction for the agent.

    Notes:
        - Providers and models must be registered in ModelRegistry before use.
        - `generate_response()` is asynchronous and should be awaited (e.g., in Jupyter).
        - Errors are raised if the provider or model cannot be found.
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
        """Normalize provider name to lowercase, defaulting to 'openai'."""
        return provider.lower() if provider else "openai"

    def _validate_provider(self, provider_key: str):
        """Raise error if provider is not registered in ModelRegistry."""
        if provider_key not in ModelRegistry.list_providers():
            raise ValueError(
                f"Unsupported provider '{provider_key}'. "
                f"Available providers: {ModelRegistry.list_providers()}. Make sure you have registered all providers if the list is empty."
            )

    def _get_model(self, provider_key: str, model_name: str) -> Any:
        """Retrieve model from ModelRegistry or raise error if not found."""
        self._validate_provider(provider_key)
        model = ModelRegistry.get_model(provider_key, model_name)
        if model is None:
            raise ValueError(
                f"Model '{model_name}' not found in provider '{provider_key}'. "
                f"Available models: {ModelRegistry.list_models(provider_key)}"
            )
        return model

    def _create_agent(self, model: Any, system_prompt: Optional[str]) -> Agent:
        """Wrap model in a pydantic_ai.Agent with optional system prompt."""
        return Agent(model=model, system_prompt=system_prompt)

    async def generate_response(self, prompt: Union[str, Sequence[UserContent], None]) -> str:
        """
        Run the agent with a user prompt.

        Parameters:
        prompt (str | Sequence[UserContent] | None): The user prompt. Can be a plain string, a structured sequence of UserContent, or None.

        Returns
        str
            The agent's output (default output type is string).

         Notes:
        - This method is safe to call concurrently (e.g. via asyncio.gather).
        - Rate limits and API constraints depend on the provider.
        """
        response = await self.agent.run(prompt)
        return response