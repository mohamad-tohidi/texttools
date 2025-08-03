from typing import Dict, List, Literal


class Gemma3Formatter:
    """
    Formatter that merges consecutive user messages (strings) with '\n'
    and leaves assistant messages alone. No image‐handling, no extra tokens.
    """

    ROLE = "role"
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"
    CONTENT = "content"
    VALID_ROLES = {USER_ROLE, ASSISTANT_ROLE}

    def format(
        self, messages: List[Dict[Literal["role", "content"], str]]
    ) -> List[Dict[str, str]]:
        """
        :param messages: list of {"role": ..., "content": ...}, where role is "user", "assistant", or "system"
        :return: a new list where consecutive "user" messages are merged into single entries
        """

        merged: List[Dict[str, str]] = []

        for msg in messages:
            role, content = msg[self.ROLE], msg[self.CONTENT].strip()

            # Replace "system" role with "user" role
            if role == "system":
                role = self.USER_ROLE

            # Raise value error if msg["role"] wan't a valid role
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
