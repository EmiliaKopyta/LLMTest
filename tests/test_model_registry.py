from __future__ import annotations
import pytest
from lib.registry.ModelRegistry import ModelRegistry


# Fixtures (automatically runs before every test)

@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure ModelRegistry is empty before each test."""
    ModelRegistry._registry.clear()
    yield
    ModelRegistry._registry.clear()

# Fake provider configs

class FakeProviderOK:
    """Fake provider configuration that registers a single model successfully."""
    provider_name = "testprovider"

    def register_models(self):
        ModelRegistry.register_model(self.provider_name, "model-a", object())

class FakeProviderFail:
    """Fake provider configuration that raises an exception during registration."""
    provider_name = "failprovider"

    def register_models(self):
        raise RuntimeError("registration failed")

# Unit tests: core

def test_register_model_and_get_model():
    """Verify that a model can be registered and retrieved from the registry."""
    model_obj = object()
    ModelRegistry.register_model("OpenAI", "gpt-x", model_obj)

    assert ModelRegistry.get_model("openai", "gpt-x") is model_obj
    assert ModelRegistry.get_model("OPENAI", "gpt-x") is model_obj

def test_list_providers_and_models():
    """Verify that listing providers and models returns expected values."""
    ModelRegistry.register_model("openai", "m1", object())
    ModelRegistry.register_model("openai", "m2", object())
    ModelRegistry.register_model("anthropic", "c1", object())

    providers = ModelRegistry.list_providers()
    assert set(providers) == {"openai", "anthropic"}

    assert set(ModelRegistry.list_models("openai")) == {"m1", "m2"}
    assert set(ModelRegistry.list_models("anthropic")) == {"c1"}

# Unit tests: add model

def test_add_model_requires_existing_provider():
    """Verify that add_model() fails if provider is not registered. """
    with pytest.raises(ValueError):
        ModelRegistry.add_model("missing", "m1", object())

def test_add_model_rejects_duplicates():
    """Verify that add_model() fails when the model already exists under the provider."""
    ModelRegistry.register_model("openai", "m1", object())

    with pytest.raises(ValueError):
        ModelRegistry.add_model("openai", "m1", object())

def test_add_model_success():
    """Verify that add_model() registers a new model for an existing provider."""
    ModelRegistry.register_model("openai", "m1", object())
    new_model = object()

    ModelRegistry.add_model("openai", "m2", new_model)

    assert ModelRegistry.get_model("openai", "m2") is new_model

# Unit tests: remove model

def test_remove_model_removes_existing():
    """Verify that remove_model() removes a model if it exists."""
    ModelRegistry.register_model("openai", "m1", object())
    assert "m1" in ModelRegistry.list_models("openai")

    ModelRegistry.remove_model("openai", "m1")

    assert "m1" not in ModelRegistry.list_models("openai")

def test_remove_model_is_safe_for_missing_model():
    """Verify that remove_model() does not raise errors for missing entries."""
    ModelRegistry.remove_model("openai", "does-not-exist")

# Unit tests: remove provider

def test_remove_provider_removes_all_models():
    """
    Verify that remove_provider() deletes the provider and all its models.
    """
    ModelRegistry.register_model("openai", "m1", object())
    ModelRegistry.register_model("openai", "m2", object())

    ModelRegistry.remove_provider("openai")

    assert "openai" not in ModelRegistry.list_providers()
    assert ModelRegistry.get_model("openai", "m1") is None

def test_remove_provider_is_safe_when_missing():
    """
    Verify that remove_provider() does not crash when provider is not registered.
    """
    ModelRegistry.remove_provider("missing-provider")

# Unit tests: register provider safety

def test_register_provider_skips_if_api_key_missing(monkeypatch):
    """Verify that register_provider() skips registration if the provider API key is missing."""
    # do NOT set TESTPROVIDER_API_KEY
    provider = FakeProviderOK()

    ModelRegistry.register_provider(provider)

    assert "testprovider" not in ModelRegistry.list_providers()
    assert ModelRegistry.get_model("testprovider", "model-a") is None

def test_register_provider_registers_when_key_present(monkeypatch):
    """Verify that register_provider() registers models when the API key exists."""
    monkeypatch.setenv("TESTPROVIDER_API_KEY", "dummy")

    provider = FakeProviderOK()
    ModelRegistry.register_provider(provider)

    assert "testprovider" in ModelRegistry.list_providers()
    assert "model-a" in ModelRegistry.list_models("testprovider")

def test_register_provider_raises_if_provider_already_registered(monkeypatch):
    """Verify that attempting to register an already registered provider raises ValueError."""
    monkeypatch.setenv("TESTPROVIDER_API_KEY", "dummy")

    provider = FakeProviderOK()
    ModelRegistry.register_provider(provider)

    with pytest.raises(ValueError):
        ModelRegistry.register_provider(provider)

def test_register_provider_skips_if_provider_register_models_fails(monkeypatch):
    """Verify that provider registration is skipped if provider.register_models() raises an exception."""
    monkeypatch.setenv("FAILPROVIDER_API_KEY", "dummy")

    provider = FakeProviderFail()
    ModelRegistry.register_provider(provider)

    assert "failprovider" not in ModelRegistry.list_providers()