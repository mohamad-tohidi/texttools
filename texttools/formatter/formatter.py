from texttools.formatter.base_formatter import BaseFormatter


class UserMergeFormatter(BaseFormatter):
    """
    Formatter that merges consecutive user messages (strings) with blank line.
    It replaces system role with user role and doesn't change assistant messages.
    """

    def _validate_input(self, messages: list[dict[str, str]]):
        valid_keys = {"role", "content"}
        valid_roles = {"user", "assistant"}

        for message in messages:
            # Validate keys
            if set(message.keys()) != valid_keys:
                raise ValueError(
                    f"Message dict keys must be exactly {valid_keys}, got {set(message.keys())}"
                )
            # Validate roles
            role = message["role"]
            if role != "system" and role not in valid_roles:
                raise ValueError(f"Unexpected role: {role}")

    def format(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        self._validate_input(messages)

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
