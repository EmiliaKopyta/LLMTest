from abc import ABC, abstractmethod
from typing import Any

class BaseProviderConfig(ABC):
    """
    Abstract base class for provider configuration.

    Subclasses must define:
    - `provider_name`: a property returning the provider's name.
    - `get_models()`: a method returning a dictionary of available models.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    def get_models(self) -> dict[str, Any]:
        pass