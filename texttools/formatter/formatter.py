from texttools.formatter.base import BaseFormatter


class Formatter(BaseFormatter):
    """
    Formatter that merges consecutive user messages (strings) with '\n'
    and leaves assistant messages alone. No image‐handling, no extra tokens.
    """

    ROLE = "role"
    CONTENT = "content"
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"
    VALID_ROLES = {USER_ROLE, ASSISTANT_ROLE}
    VALID_KEYS = {ROLE, CONTENT}

    def format(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        :param messages: list of {"role": ..., "content": ...}, where role is "user", "assistant", or "system"
        :return: a new list where consecutive "user" messages are merged into single entries
        """

        merged: list[dict[str, str]] = []

        for message in messages:
            # Validate keys strictly
            if set(message.keys()) != self.VALID_KEYS:
                raise ValueError(
                    f"Message dict keys must be exactly {self.VALID_KEYS}, got {set(message.keys())}"
                )

            role, content = message[self.ROLE], message[self.CONTENT].strip()

            # Replace "system" role with "user" role
            if role == "system":
                role = self.USER_ROLE

            # Raise value error if message["role"] wan't a valid role
            if role not in self.VALID_ROLES:
                raise ValueError(f"Unexpected role: {role}")

            # Merge with previous user turn
            if (
                merged
                and role == self.USER_ROLE
                and merged[-1][self.ROLE] == self.USER_ROLE
            ):
                merged[-1][self.CONTENT] += "\n" + content

            # Otherwise, start a new turn
            else:
                merged.append({self.ROLE: role, self.CONTENT: content})

        return merged
