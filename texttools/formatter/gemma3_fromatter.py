from typing import Dict, List, Literal


class Gemma3Formatter:
    """
    Formatter that merges consecutive user messages (strings) with '\n'
    and leaves assistant messages alone. No image‐handling, no extra tokens.
    """

    def format(
        self,
        messages: List[Dict[Literal["role", "content"], str]]
    ) -> List[Dict[str, str]]:
        """
        :param messages: a list of {"role": "user"|"assistant", "content": <string>}
        :return: a new list where consecutive "user" messages have been merged.
        """
        merged: List[Dict[str, str]] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()

            if merged and role == "user" and merged[-1]["role"] == "user":
                # Merge with previous user turn
                merged[-1]["content"] += "\n" + content
            else:
                # Start a new turn
                merged.append({"role": role, "content": content})

        return merged
