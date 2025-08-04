from typing import Any, Optional

from openai import OpenAI
from pydantic import BaseModel

from texttools.base.base_keyword_extractor import BaseKeywordExtractor
from texttools.formatter import Gemma3Formatter


class Output(BaseModel):
    keywords: list


class GemmaKeywordExtractor(BaseKeywordExtractor):
    """
    Keyword extractor for Gemma-style models with optional reasoning step.
    Outputs JSON with a single array field: {"keywords": ["keyword1", "keyword2", ...]}.

    Allows optional extra instructions via `prompt_template`.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        use_reason: bool = False,
        chat_formatter: Optional[Any] = None,
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

        self.output = Output

    def _build_messages(
        self, text: str, reason: Optional[str] = None
    ) -> list[dict[str, str]]:
        clean_text = self.preprocess(text)

        messages: list[dict[str, str]] = []

        if self.prompt_template:
            messages.append({"role": "user", "content": self.prompt_template})

        if reason:  # Include the reason if available
            messages.append(
                {"role": "user", "content": f"Based on this analysis: {reason}"}
            )

        messages.append(
            {
                "role": "user",
                "content": "Extract the most relevant keywords from the following text. Provide them as a list of strings.",
            }
        )
        messages.append({"role": "user", "content": clean_text})

        # Ensure the schema is dumped as a valid JSON string
        schema_instr = f"Respond only in JSON format: {self.output.model_dump_json()}"
        messages.append({"role": "user", "content": schema_instr})

        # Deprecated
        # messages.append(
        #     {"role": "assistant", "content": "{"}
        # )  # Start with '{' to hint JSON

        messages = self.chat_formatter.format(messages=messages)

        return messages

    def _reason(self, text: str) -> str:
        """
        Internal reasoning step to help the model identify potential keywords.
        """
        messages = [
            {
                "role": "user",
                "content": """
                    Analyze the following text to identify its main topics, concepts, and important terms.
                    Provide a concise summary of your findings that will help in extracting relevant keywords.
                    """,
            },
            {
                "role": "user",
                "content": f"""
                    {text}
                    """,
            },
        ]
        messages = self.chat_formatter.format(messages=messages)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )

        reason_summary = resp.choices[0].message.content.strip()
        return reason_summary

    def extract_keywords(self, text: str) -> list[str]:
        """
        Extracts keywords from `text`.
        Optionally uses an internal reasoning step for better accuracy.
        """
        reason_summary = None
        if self.use_reason:
            reason_summary = self._reason(text)

        messages = self._build_messages(text, reason_summary)

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=Output,
            temperature=self.temperature,
            extra_body=dict(guided_decoding_backend="auto"),
            **self.client_kwargs,
        )

        message = completion.choices[0].message

        keywords = message.parsed.keywords

        # dispatch and return
        self._dispatch({"original_text": text, "keywords": keywords})
        return keywords
