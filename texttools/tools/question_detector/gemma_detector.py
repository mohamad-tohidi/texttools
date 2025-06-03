from typing import Any, Dict, List, Optional
import json
from openai import OpenAI
from texttools.base.base_question_detector import BaseQuestionDetector
from texttools.formatter import Gemma3Formatter

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
        chat_formatter: Optional[Any] = None,
        use_reason: bool = False,
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

        self.chat_formatter = chat_formatter or Gemma3Formatter(
            add_generation_prompt=True
        )

        self.use_reason = use_reason
        self.prompt_template = prompt_template
        
        self.json_schema = {
            "is_question": bool
            }

    def _build_messages(self, text: str, reason: str = None) -> List[Dict[str, str]]:
        clean = self.preprocess(text)
        schema_instr = f'respond only in JSON format: {self.json_schema}'
        messages: List[Dict[str, str]] = []

        if reason:
            messages.append({"role": "user", "content": reason})

        messages.append({"role": "user", "content": schema_instr})
        if self.prompt_template:
            messages.append({"role": "user", "content": self.prompt_template})
        messages.append({"role": "user", "content": clean})
        messages.append({"role": "assistant", "content": "{\n"})
        
        # this line will restructure the messages
        # based on the formatter that we provided
        # some models will require custom settings
        restructured = self.chat_formatter.format(messages=messages)
        
        
        return restructured

    def _reason(self, text: str) -> list:
        messages = [
            {
                "role": "user",
                "content": 
                    """
                    we want to analyze this text snippet to see if it contains any question
                    or request of some kind or not
                    read the text, and reason about it being a request or not
                    summerized
                    short answer
                    """},
            {
                "role": "user",
                "content":
                    f"""
                    {text}
                    """
            }
        ]
        
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        
        reason = resp.choices[0].message.content.strip()
        return reason

    def detect(self, text: str) -> bool:
        """
        Returns True if `text` is a question, False otherwise.
        Optionally uses an internal reasoning step for better accuracy.
        """
        reason_summary = None
        if self.use_reason:
            reason_summary = self._reason(text)

        
        # print(reason_summary)
        
        messages = self._build_messages(text, reason_summary)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        raw = resp.choices[0].message.content.strip()

        if not raw.startswith("{"):
            raw = "{" + raw
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nRaw output: {raw}")

        result = parsed.get("is_question")
        if not isinstance(result, bool):
            raise ValueError(f"Invalid response schema, got: {parsed}")

        # dispatch and return
        self._dispatch({"question": text, "result": result})
        return result
