from texttools.formatters.base_formatter import BaseFormatter


class UserMergeFormatter(BaseFormatter):
    """
    Merges consecutive user messages into a single message, separated by newlines.

    This is useful for condensing a multi-turn user input into a single coherent
    message for the LLM. Assistant and system messages are left unchanged and
    act as separators between user message groups.

    Raises:
        ValueError: If the input messages have invalid structure or roles.
    """

    def format(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        merged: list[dict[str, str]] = []

        for message in messages:
            role, content = message["role"], message["content"].strip()

            # Merge with previous user turn
            if merged and role == "user" and merged[-1]["role"] == "user":
                merged[-1]["content"] += "\n" + content

            # Otherwise, start a new turn
            else:
                merged.append({"role": role, "content": content})

        return merged
