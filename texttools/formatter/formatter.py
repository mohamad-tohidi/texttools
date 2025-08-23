from texttools.formatter.base_formatter import BaseFormatter


class Formatter(BaseFormatter):
    """
    Formatter that merges consecutive user messages (strings) with blank line.
    It leaves assistant messages alone.
    """

    VALID_ROLES = {"user", "assistant"}
    VALID_KEYS = {"role", "content"}

    def format(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        merged: list[dict[str, str]] = []

        for message in messages:
            # Validate keys strictly
            if set(message.keys()) != self.VALID_KEYS:
                raise ValueError(
                    f"Message dict keys must be exactly {self.VALID_KEYS}, got {set(message.keys())}"
                )

            role, content = message["role"], message["content"].strip()

            # Replace "system" role with "user" role
            if role == "system":
                role = "user"

            # Raise value error if message["role"] wan't a valid role
            if role not in self.VALID_ROLES:
                raise ValueError(f"Unexpected role: {role}")

            # Merge with previous user turn
            if merged and role == "user" and merged[-1]["role"] == "user":
                merged[-1]["content"] += "\n" + content

            # Otherwise, start a new turn
            else:
                merged.append({"role": role, "content": content})

        return merged
