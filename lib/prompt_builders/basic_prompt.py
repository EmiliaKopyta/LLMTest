def basic_prompt(row, col: str = "prompt") -> str:
    """
    Build a simple prompt from a single column in the dataset.

    This function is intended for tasks where the dataset already contains
    a ready-made prompt string (e.g., a question or instruction) and the model
    should be asked to provide an answer. It extracts the text from the given
    column and appends a standard "Answer:" suffix to clearly indicate where
    the model should respond.

    Parameters:
    row (dict): A single dataset record containing the prompt text.
    col (str):  The column name in the dataset that holds the prompt text, defaults to "prompt".

    Returns:
    str
        A formatted string consisting of the prompt text followed by "Answer:".
    """

    return f"{row[col]}\nAnswer:"