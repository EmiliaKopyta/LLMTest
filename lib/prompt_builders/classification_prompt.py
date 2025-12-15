def classification_prompt(row, text_column="text", label_column="label", labels=None):
    """
    Build a generic prompt for text classification tasks.

    This function takes a dataset row containing a text sample and its label,
    and constructs a multiple-choice style prompt with the provided labels.
    The model is instructed to reply with exactly one label from the list.

    Parameters:
    row (dict): A single dataset record containing text and label fields.
    text_column (str): Name of the column holding the input text, defaults to "text".
    label_column (str): Name of the column holding the true label. defaults to "label".
    labels (list[str], optional): List of possible labels. If None, defaults to ["0", "1"].

    Returns:
    dict
        A dictionary with keys:
        - "question": the classification prompt text
        - "choices": the list of labels
        - "answer": the true label from the dataset
    """

    text = row[text_column]
    if labels is None:
        labels = ["0", "1"]

    return {
        "question": (
            f"Classify the following text into one of the given labels:\n\n"
            f"{text}\n\n"
            f"Labels: {labels}\n\n"
            f"Reply with exactly one label from the list."
        ),
        "choices": labels,
        "answer": row[label_column],
    }