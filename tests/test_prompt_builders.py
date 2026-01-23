from __future__ import annotations
import pytest
from lib.prompt_builders.basic_prompt import basic_prompt
from lib.prompt_builders.classification_prompt import classification_prompt
from lib.prompt_builders.provide_choices_prompt import provide_choices_prompt

# Unit tests: basic prompt builder

def test_basic_prompt_uses_default_column_and_suffix():
    """Verify that the builder reads the default 'prompt' column and appends 'Answer:'."""
    row = {"prompt": "What is 2+2?"}
    out = basic_prompt(row)

    assert out == "What is 2+2?\nAnswer:"

def test_basic_prompt_supports_custom_column():
    """Verify that a custom column name can be used to build the prompt."""
    row = {"question": "Explain Newton's method."}
    out = basic_prompt(row, col="question")

    assert out == "Explain Newton's method.\nAnswer:"

def test_basic_prompt_raises_keyerror_for_missing_column():
    """Verify that missing prompt column raises KeyError (current contract)."""
    row = {"text": "hello"}
    with pytest.raises(KeyError):
        basic_prompt(row) # default col='prompt'

# Unit tests: provide choices prompt builder

def test_provide_choices_prompt_formats_options_from_list():
    """Verify that choices provided as a list are formatted into A/B/C options."""
    row = {
        "question": "What is the capital of France?",
        "choices": ["Paris", "Berlin", "Rome"],
    }

    out = provide_choices_prompt(row)

    assert out.startswith("What is the capital of France?\nOptions:\n")
    assert "A. Paris" in out
    assert "B. Berlin" in out
    assert "C. Rome" in out
    assert out.endswith("\nAnswer:")

def test_provide_choices_prompt_parses_choices_from_string_list_repr():
    """Verify that choices provided as a string representation of a Python list are parsed correctly."""
    row = {
        "question": "Choose one:",
        "choices": "['Ala', 'Ola', 'Ela']",
    }

    out = provide_choices_prompt(row)

    assert "A. Ala" in out
    assert "B. Ola" in out
    assert "C. Ela" in out
    assert out.endswith("\nAnswer:")

def test_provide_choices_prompt_fallback_parsing_when_literal_eval_fails():
    """Verify that fallback parsing works when literal_eval fails for malformed strings."""
    row = {
        "question": "Pick one:",
        "choices": "['Red' 'Green' 'Blue']",
    }

    out = provide_choices_prompt(row)

    assert "Options:" in out
    assert "A." in out
    assert out.endswith("\nAnswer:")

def test_provide_choices_prompt_raises_keyerror_when_missing_columns():
    """Verify that missing required fields raise KeyError."""
    row = {"choices": ["x", "y"]}

    with pytest.raises(KeyError):
        provide_choices_prompt(row)

def test_provide_choices_prompt_supports_custom_choices_column():
    """Verify that builder supports a custom choices column name."""
    row = {"question": "Pick:", "options": ["x", "y"]}

    out = provide_choices_prompt(row, c_col="options")

    assert "A. x" in out
    assert "B. y" in out

# Unit tests: classification prompt builder

def test_classification_prompt_returns_expected_structure_default_labels():
    """Verify that classification_prompt returns a dict with question/choices/answer and default labels."""
    row = {"text": "I love this movie!", "label": "1"}

    out = classification_prompt(row)

    assert isinstance(out, dict)
    assert set(out.keys()) == {"question", "choices", "answer"}

    assert "Classify the following text" in out["question"]
    assert "I love this movie!" in out["question"]

    assert out["choices"] == ["0", "1"]
    assert out["answer"] == "1"

def test_classification_prompt_supports_custom_labels():
    """Verify that classification_prompt supports custom label lists."""
    row = {"text": "This was terrible.", "label": "negative"}

    out = classification_prompt(row, labels=["positive", "negative"])

    assert out["choices"] == ["positive", "negative"]
    assert out["answer"] == "negative"
    assert "Labels: ['positive', 'negative']" in out["question"]

def test_classification_prompt_supports_custom_column_names():
    """Verify that classification_prompt supports custom column names for text and label."""
    row = {"content": "Neutral statement.", "gold": "neutral"}

    out = classification_prompt(
        row,
        text_column="content",
        label_column="gold",
        labels=["neutral", "other"]
    )

    assert out["choices"] == ["neutral", "other"]
    assert out["answer"] == "neutral"
    assert "Neutral statement." in out["question"]

def test_classification_prompt_raises_keyerror_when_missing_text():
    """Verify that missing text column raises KeyError."""
    row = {"label": "1"}

    with pytest.raises(KeyError):
        classification_prompt(row)

def test_classification_prompt_raises_keyerror_when_missing_label():
    """Verify that missing label column raises KeyError."""
    row = {"text": "Hello"}

    with pytest.raises(KeyError):
        classification_prompt(row)