# test_build_messages.py

import json
from typing import Any, Dict, List, Optional, Union

# ────────────────────────────────────────────────────────────────────────────────
# (1) Gemma3Formatter: exactly as in your code, enforcing user/assistant alternation.
# ────────────────────────────────────────────────────────────────────────────────

from typing import Any, Dict, List, Union

# texttools/formatter.py

from typing import Any, Dict, List, Union


class Gemma3Formatter:
    """
    Formatter that merges consecutive user messages (strings) with '\n'
    and leaves assistant messages alone. No image‐handling, no extra tokens.
    """

    def format(
        self,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        :param messages: a list of {"role": "user"|"assistant", "content": <string>}
        :return: a new list where consecutive "user" messages have been merged.
        """
        merged: List[Dict[str, str]] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()

            if not merged:
                # First message: just append
                merged.append({"role": role, "content": content})
            else:
                last = merged[-1]
                if role == "user" and last["role"] == "user":
                    # Merge with previous user turn
                    last["content"] += "\n" + content
                else:
                    # Otherwise, start a new turn
                    merged.append({"role": role, "content": content})

        return merged


# ────────────────────────────────────────────────────────────────────────────────
# (2) GemmaQuestionDetector with only _build_messages (preprocess = identity).
# ────────────────────────────────────────────────────────────────────────────────

class GemmaQuestionDetector:
    def __init__(
        self,
        model: str,
        chat_formatter: Optional[Any] = None,
        prompt_template: Optional[str] = None,
    ):
        self.model = model
        self.chat_formatter = chat_formatter or Gemma3Formatter()
        self.prompt_template = prompt_template

    def preprocess(self, text: str) -> str:
        # Stub: simply trim whitespace
        return text.strip()

    def _build_messages(self, text: str, reason: Optional[str] = None) -> List[Dict[str, str]]:
        clean = self.preprocess(text)
        schema_instr = f'respond only in JSON format: {{"is_question": bool}}'
        messages: List[Dict[str, str]] = []

        if reason:
            messages.append({"role": "user", "content": reason})

        messages.append({"role": "user", "content": schema_instr})

        if self.prompt_template:
            messages.append({"role": "user", "content": self.prompt_template})

        messages.append({"role": "user", "content": clean})
        messages.append({"role": "assistant", "content": "{\n"})

        restructured = self.chat_formatter.format(messages=messages)
        return restructured


# ────────────────────────────────────────────────────────────────────────────────
# (3) Test function that calls _build_messages and prints the result or exception
# ────────────────────────────────────────────────────────────────────────────────

def test_build_messages():
    print("\n==== Test: No reason, no prompt_template ====")
    detector1 = GemmaQuestionDetector(model="gemma-3")
    try:
        msgs1 = detector1._build_messages("Hello, is this working?", reason=None)
        print(json.dumps(msgs1, indent=2))
    except Exception as e:
        print("Error:", e)

    print("\n==== Test: With prompt_template, no reason ====")
    detector2 = GemmaQuestionDetector(model="gemma-3", prompt_template="Please check if this is a question.")
    try:
        msgs2 = detector2._build_messages("What's your name?", reason=None)
        print(json.dumps(msgs2, indent=2))
    except Exception as e:
        print("Error:", e)

    print("\n==== Test: With reason and prompt_template ====")
    detector3 = GemmaQuestionDetector(model="gemma-3", prompt_template="Check it carefully.")
    try:
        msgs3 = detector3._build_messages("Is Python your favorite language?", reason="I need to be sure this is a question.")
        print(json.dumps(msgs3, indent=2))
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    test_build_messages()
