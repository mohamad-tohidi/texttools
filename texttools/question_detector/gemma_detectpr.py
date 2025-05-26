from typing import Any, Dict, List
import json
from openai import OpenAI
from texttools.base.base_question_detector import BaseQuestionDetector


class GemmaQuestionDetector(BaseQuestionDetector):
    """
    Simplified binary question detector for Gemma-style models without system prompts.
    Outputs JSON with a single boolean field: {"is_question": true|false}.

    Allows optional extra instructions via `prompt_template`.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        temperature: float = 0.0,
        prompt_template: str = None,
        handlers: List[Any] = None,
        **client_kwargs: Any
    ):
        super().__init__(handlers)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs

        self.prompt_text = prompt_template
        
        
        self.json_schema = {
            "is_question": bool
            }


    def _build_messages(self, text: str) -> List[Dict[str, str]]:
        clean = self.preprocess(text)
        schema_instruction = f'respond only in JSON format, in this form {self.json_schema}'
        messages: List[Dict[str, str]] = [
            {"role": "user", "content": schema_instruction},
        ]
        if self.prompt_text:
            messages.append({"role": "user", "content": self.prompt_text})
        messages.append({"role": "user", "content": clean})
        messages.append({"role": "assistant", "content": "{"})
        return messages

    def detect(self, text: str) -> bool:
        messages = self._build_messages(text)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        raw = resp.choices[0].message.content.strip()

        try:
            if not raw.startswith("{"):
                raw = "{" + raw
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nRaw output: {raw}")

        if "is_question" not in parsed or not isinstance(parsed["is_question"], bool):
            raise ValueError(f"Invalid response schema, got: {parsed}")

        result = parsed["is_question"]
        self._dispatch({"question": text, "result": result})
        return result