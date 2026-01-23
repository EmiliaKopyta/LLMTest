from __future__ import annotations
import pytest
import asyncio
from lib.registry.ModelRegistry import ModelRegistry
from lib.PromptHandler import PromptHandler

# NOTE:
# These tests focus on PromptHandler as a wrapper for pydantic_ai.AI Agent.
# The real Agent is patched in unit tests to keep them deterministic
# and independent from external libraries or network/API calls.
# A single optional integration test checks compatibility with the real Agent.

# Fixtures (runs automatically)

@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure ModelRegistry is empty for each test."""
    ModelRegistry._registry.clear()
    yield
    ModelRegistry._registry.clear()

class FakeAgent:
    """Minimal fake Agent replacement used for deterministic tests. Captures init arguments and returns a fixed response from 'run()'."""
    def __init__(self, model=None, system_prompt=None):
        self.model = model
        self.system_prompt = system_prompt

    async def run(self, prompt):
        return "FAKE_RESPONSE"

@pytest.fixture(autouse=True)
def _patch_agent(monkeypatch):
    """Replace pydantic_ai.Agent with FakeAgent for all tests in this module."""
    monkeypatch.setattr("lib.PromptHandler.Agent", FakeAgent)

# Unit tests: normalize

def test_normalize_provider_default_openai():
    """Verify that provider defaults to 'openai' when None is given."""
    ModelRegistry.register_model("openai", "m1", object())
    handler = PromptHandler(model_name="m1", provider=None)

    assert handler.provider_key == "openai"

def test_normalize_provider_lowercases():
    """Verify that provider is normalized to lowercase."""
    ModelRegistry.register_model("openai", "m1", object())
    handler = PromptHandler(model_name="m1", provider="OpenAI")

    assert handler.provider_key == "openai"

# Unit tests: errors

def test_validate_provider_raises_if_missing():
    """Verify that an unknown provider raises ValueError."""
    with pytest.raises(ValueError):
        PromptHandler(model_name="m1", provider="missing")

def test_get_model_raises_if_missing_model():
    """Verify that missing model name raises ValueError under existing provider."""
    ModelRegistry.register_model("openai", "existing", object())

    with pytest.raises(ValueError):
        PromptHandler(model_name="missing_model", provider="openai")

# Unit tests: Agent usage (patched)

def test_create_agent_receives_system_prompt():
    """Verify that PromptHandler passes system_prompt to Agent constructor."""
    ModelRegistry.register_model("openai", "m1", object())
    handler = PromptHandler(model_name="m1", provider="openai", system_prompt="SYSTEM_TEST")

    assert isinstance(handler.agent, FakeAgent)
    assert handler.agent.system_prompt == "SYSTEM_TEST"

def test_generate_response_returns_agent_output():
    """Verify that generate_response() returns the value produced by Agent.run()."""
    ModelRegistry.register_model("openai", "m1", object())
    handler = PromptHandler(model_name="m1", provider="openai")

    out = asyncio.run(handler.generate_response("hello"))

    assert out == "FAKE_RESPONSE"

# Integration test: pydantic_ai.Agent

@pytest.mark.integration
def test_integration_prompt_handler_with_real_agent(monkeypatch):
    """
    Verify that PromptHandler is compatible with pydantic_ai.Agent.
    This test is skipped if Agent cannot be used in the current environment.
    It does NOT perform any real API calls.
    """
    try:
        from pydantic_ai import Agent as RealAgent
    except Exception as e:
        pytest.skip(f"pydantic_ai.Agent is not available: {e}")

    monkeypatch.setattr("lib.PromptHandler.Agent", RealAgent)

    # pydantic_ai supports a built-in offline test model by passing model="test".
    ModelRegistry.register_model("openai", "m1", "test")

    handler = PromptHandler(model_name="m1", provider="openai")

    try:
        result = asyncio.run(handler.generate_response("hello"))
    except Exception as e:
        pytest.skip(f"Real Agent could not run in this environment: {e}")

    # AgentRunResult should contain an '.output' attribute (later used in TestRunner)
    assert hasattr(result, "output")
    assert isinstance(result.output, str)