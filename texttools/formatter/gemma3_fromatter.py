from typing import Any, Dict, List, Union


class Gemma3Formatter:
    """
    Takes a list of messages, each of the form
        {"role": "system"|"user"|"assistant", "content": <str> or <List[{"type":..., "text":...}]>},
    and turns it into a new list of messages that follow the exact
    <start_of_turn>…<end_of_turn> markup logic in your Jinja template.

    Usage:
        formatter = Gemma3Formatter(add_generation_prompt=True)
        out_msgs = formatter.format(messages_in)
        # out_msgs is List[{"role": "user"|"model", "content": "..."}]
    """

    def __init__(self, *, add_generation_prompt: bool = False):
        self.add_generation_prompt = add_generation_prompt

    def format(
        self,
        messages: List[Dict[str, Union[str, List[Dict[str,str]]]]]
    ) -> List[Dict[str, str]]:
        """
        :param messages: A list of dicts, each with:
            - "role":   one of "system", "user", or "assistant"
            - "content": either
                 • a plain Python string, or
                 • a List[{"type": "text" | "image", "text": <str>}]
        :returns: A new List[{"role": "user"|"model", "content": "<start_of_turn>…<end_of_turn>\n"}].
        """

        # 1) Check if the very first message is a "system" block
        #    If so, capture its text (or first text‐item) as first_user_prefix,
        #    then drop it from the main “loop_messages” list.
        first_user_prefix: str
        loop_messages: List[Dict[str, Any]]

        if len(messages) > 0 and messages[0]["role"] == "system":
            system_content = messages[0]["content"]
            if isinstance(system_content, str):
                first_user_prefix = system_content.strip() + "\n\n"
            else:
                # “content” is iterable; pick the first text‐chunk
                # (exactly as Jinja does: system_content[0]["text"])
                first_user_prefix = system_content[0]["text"].strip() + "\n\n"
            loop_messages = messages[1:]
        else:
            first_user_prefix = ""
            loop_messages = messages

        # 2) Iterate through loop_messages, enforce alternation, rename 'assistant'→'model'
        restructured: List[Dict[str, str]] = []

        for idx, msg in enumerate(loop_messages):
            # a) Check alternating roles: user, assistant, user, assistant, ...
            wants_user_role = (idx % 2 == 0)
            is_user_role = (msg["role"] == "user")
            if is_user_role != wants_user_role:
                raise ValueError(
                    f"Conversation roles must alternate user/assistant/user/... "
                    f"but got index={idx}, role={msg['role']!r}."
                )

            # b) Map "assistant" → "model"
            role = "model" if msg["role"] == "assistant" else msg["role"]  # either "user" or "model"

            # c) Build a single trimmed string out of msg["content"]:
            raw_content = msg["content"]
            if isinstance(raw_content, str):
                body = raw_content.strip()
            else:
                # raw_content is iterable of {"type":..., "text": ...} entries
                pieces: List[str] = []
                for item in raw_content:
                    if item.get("type") == "image":
                        pieces.append("<start_of_image>")
                    elif item.get("type") == "text":
                        pieces.append(item["text"].strip())
                    else:
                        raise ValueError(f"Invalid content‐item: {item!r}")
                body = "".join(pieces)

            # d) Prepend first_user_prefix only on the very first iteration
            if idx == 0 and first_user_prefix:
                body = first_user_prefix + body

            # e) Wrap with <start_of_turn>…<end_of_turn>\n
            wrapped = f"<start_of_turn>{role}\n{body}<end_of_turn>\n"

            restructured.append({"role": role, "content": wrapped})

        # 3) Optionally append the “generation prompt” token:
        if self.add_generation_prompt:
            # This is literally: <start_of_turn>model\n
            restructured.append({"role": "model", "content": "<start_of_turn>model\n"})

        return restructured
