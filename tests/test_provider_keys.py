from __future__ import annotations
import pytest
from lib.config.ProviderKeys import ProviderKeys

# Unit tests: key collection

def test_providerkeys_collects_only_suffix_api_key(monkeypatch: pytest.MonkeyPatch):
    """Verify that ProviderKeys collects only variables ending with '_API_KEY'."""
    monkeypatch.setenv("TEST_PROVIDER_API_KEY", "dummy_value")
    monkeypatch.setenv("NOT_A_KEY", "should_be_ignored")
    monkeypatch.setenv("API_KEY", "should_be_ignored")

    pk = ProviderKeys()

    assert "TEST_PROVIDER_API_KEY" in pk.keys
    assert pk.keys["TEST_PROVIDER_API_KEY"] == "dummy_value"
    assert "NOT_A_KEY" not in pk.keys
    assert "API_KEY" not in pk.keys

def test_providerkeys_collects_multiple_keys(monkeypatch: pytest.MonkeyPatch):
    """Verify that ProviderKeys can collect multiple API keys at once."""
    monkeypatch.setenv("A_API_KEY", "a123")
    monkeypatch.setenv("B_API_KEY", "b456")
    monkeypatch.setenv("C_NOT_KEY", "ignored")

    pk = ProviderKeys()

    assert pk.keys["A_API_KEY"] == "a123"
    assert pk.keys["B_API_KEY"] == "b456"
    assert "C_NOT_KEY" not in pk.keys

def test_providerkeys_preserves_original_variable_names(monkeypatch: pytest.MonkeyPatch):
    """Verify that ProviderKeys uses environment variable names as dictionary keys."""
    monkeypatch.setenv("MY_CUSTOM_PROVIDER_API_KEY", "abc")

    pk = ProviderKeys()

    assert "MY_CUSTOM_PROVIDER_API_KEY" in pk.keys
    assert pk.keys["MY_CUSTOM_PROVIDER_API_KEY"] == "abc"

def test_providerkeys_does_not_crash_without_test_keys(monkeypatch: pytest.MonkeyPatch):
    """Verify that ProviderKeys initializes correctly even if no test API keys are provided."""
    #for safety, avoids asserting 'pk.keys == {}' because there may be other *_API_KEY variables.
    pk = ProviderKeys()
    assert isinstance(pk.keys, dict)