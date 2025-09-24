from lib.providers.openai_config import OpenAIConfig
#from lib.providers.anthropic_config import AnthropicConfig  #todo

def register_all_models():
    """
    Register all models from all known providers into the ModelRegistry.
    """
    provider_configs = [
        OpenAIConfig(),
        # AnthropicConfig(),  #todo
    ]

    for config in provider_configs:
        config.register_models()

    print("All models registered in ModelRegistry")
