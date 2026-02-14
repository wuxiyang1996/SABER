"""Adversarial attack tool for the HotpotQA attack agent.

The only tool available to the attack agent is `add_suffix`, which appends
a string to the original question before it is sent to the frozen evaluation
model. The agent learns (via GRPO) what suffixes cause the eval model to
produce incorrect answers.
"""

# ---------------------------------------------------------------------------
# Tool schema (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add_suffix",
            "description": (
                "Append a suffix string to the original question. The modified "
                "question (original + suffix) will be sent to the evaluation model. "
                "Your goal is to craft a suffix that causes the evaluation model to "
                "answer the question incorrectly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "suffix": {
                        "type": "string",
                        "description": (
                            "The text to append to the end of the original question. "
                            "This could be misleading context, rephrasing, or any "
                            "adversarial text."
                        ),
                    }
                },
                "required": ["suffix"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


def apply_suffix(original_question: str, suffix: str) -> str:
    """Concatenate the original question with the adversarial suffix.

    Adds a single space between the question and suffix if the question
    doesn't already end with whitespace.
    """
    if not suffix:
        return original_question

    if original_question and not original_question[-1].isspace():
        return original_question + " " + suffix
    return original_question + suffix
