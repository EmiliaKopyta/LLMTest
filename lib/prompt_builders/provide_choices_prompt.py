import ast

def provide_choices_prompt(row, q_col: str = "question", c_col: str = "choices") -> str:
    """
    Build a multiple-choice style prompt from dataset columns.

    This function takes a dataset row containing a question and a set of possible
    answer choices, normalizes the choices into a clean list of strings, and formats
    them into a standard multiple-choice prompt. Each choice is labeled with a letter
    (A, B, C, ...) to make it clear which option the model should select.

    Parameters:
    row (dict): A single dataset record containing the question and choices.
    q_col (str): The column name holding the question text, defaults to "question".
    c_col (str): The column name holding the raw choices (list or string), defaults to "choices".

    Returns:
    str
        A formatted prompt string with the question, enumerated options, and
        an "Answer:" suffix to indicate where the model should respond.
    """
    question = row[q_col]
    raw_choices = row[c_col]

    choices = parse_choices(raw_choices)

    choices_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    return f"{question}\nOptions:\n{choices_str}\nAnswer:"

import ast

def parse_choices(raw):
    """
    Normalize the 'choices' column into a list of clean strings.

    It attempts to safely parse string representations of lists using
    `ast.literal_eval`, and falls back to manual cleaning if parsing fails.
    """
    if isinstance(raw, list):
        return [str(c).strip() for c in raw]

    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed]
        except Exception:
            pass

        cleaned = raw.strip("[]").replace("'", "").replace('"', "")
        return [c.strip() for c in cleaned.split() if c]

    return [str(raw).strip()]
