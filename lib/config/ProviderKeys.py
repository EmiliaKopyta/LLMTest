import os
from dataclasses import dataclass

@dataclass
class ProviderKeys:
    """Class for keeping track of provider keys."""
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

