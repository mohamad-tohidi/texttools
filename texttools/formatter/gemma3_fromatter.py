from typing import Dict, List, Literal


class Gemma3Formatter:
    """
    Formatter that merges consecutive user messages (strings) with '\n'
    and leaves assistant messages alone. No image‐handling, no extra tokens.
    """

    def format(
        self, messages: List[Dict[Literal["role", "content"], str]]
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
