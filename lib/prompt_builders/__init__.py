from .basic_prompt import basic_prompt
from .provide_choices_prompt import provide_choices_prompt
from .template_prompt import template_prompt

# Registry of available prompt builders
PROMPT_BUILDERS = {
    "basic_prompt": basic_prompt,
    "multiple_choice": provide_choices_prompt,
    "template": template_prompt,  # included as an example
}

def register_builder(name: str, func):
    """
    Register a new prompt builder.
    :param name: Unique name for the builder
    :param func: Callable that takes a row and returns a string prompt
    """
    if not callable(func):
        raise ValueError("Prompt builder must be callable")
    PROMPT_BUILDERS[name] = func