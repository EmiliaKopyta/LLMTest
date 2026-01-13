import os
from dataclasses import dataclass, field

@dataclass
class ProviderKeys:
    """
    Utility class for loading provider API keys from environment variables.

    Keys are discovered automatically based on the naming convention: *_API_KEY.
    All matching variables are collected into the `keys` dictionary.
    """
    keys: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        for name, value in os.environ.items():
            if name.endswith("_API_KEY"):
                self.keys[name] = value
