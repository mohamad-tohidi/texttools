from typing import Any, Optional

from openai import OpenAI
from pydantic import BaseModel

from texttools.base.base_question_detector import BaseQuestionDetector
from texttools.formatter import Gemma3Formatter


class QuestionDetection(BaseModel):
    is_question: bool


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
        handlers: list[Any] = None,
        **client_kwargs: Any,
    ):
        super().__init__(handlers)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs
        self.chat_formatter = chat_formatter or Gemma3Formatter()
        self.use_reason = use_reason
        self.prompt_template = prompt_template

    def _build_messages(self, text: str, reason: str = None) -> list[dict[str, str]]:
        clean_text = self.preprocess(text)
        messages: list[dict[str, str]] = []

        if reason:
            messages.append({"role": "user", "content": reason})

        if self.prompt_template:
            messages.append({"role": "user", "content": self.prompt_template})
        messages.append({"role": "user", "content": clean_text})

        # Restructure the messages based on the formatter; some models will require custom settings
        restructured = self.chat_formatter.format(messages=messages)

        return restructured

    def _reason(self, text: str) -> list:
        reason_prompt = f"""
                    We want to analyze this text snippet to see if it contains any question
                    or request of some kind or not.
                    Read the text, and reason about it being a request or not.
                    Summerized, Short answer
                    {text}
                    """
        messages = [
            {"role": "user", "content": reason_prompt},
        ]

        restructured = self.chat_formatter.format(messages=messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=restructured,
            temperature=self.temperature,
            **self.client_kwargs,
        )

        reason = response.choices[0].message.content.strip()
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

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=QuestionDetection,
            temperature=self.temperature,
            extra_body=dict(guided_decoding_backend="auto"),
            **self.client_kwargs,
        )
        message = completion.choices[0].message
        if message.parsed:
            result = message.parsed.is_question
        else:
            raise ValueError(
                f"Failed to parse the response. Raw content: {message.content}"
            )

        # Dispatch and return
        self._dispatch({"question": text, "result": result})
        return result
