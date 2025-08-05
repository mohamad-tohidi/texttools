import json
from typing import Any, Optional

from openai import OpenAI

from texttools.base.base_summarizer import BaseSummarizer
from texttools.handlers import ResultHandler


class GemmaSummarizer(BaseSummarizer):
    """
    Summarizer for Gemma-style models with optional reasoning step.
    Outputs JSON with a single string field: {"summary": "..."}.

    Allows optional extra instructions via `prompt_template`.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        use_reason: bool = False,
        temperature: float = 0.0,
        prompt_template: Optional[str] = None,
        handlers: Optional[list[ResultHandler]] = None,
        **client_kwargs: Any,
    ):
        super().__init__(handlers)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs

        self.use_reason = use_reason
        self.prompt_template = prompt_template

        # Define the JSON schema for the summary output
        self.json_schema = {"summary": "string"}

    def _build_messages(
        self, text: str, reason: Optional[str] = None
    ) -> list[dict[str, str]]:
        """
        Builds the message list for the LLM API call.
        """
        clean_text = self.preprocess(text)
        # Ensure the schema is dumped as a valid JSON string

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
                "content": "Please provide a concise summary of the following text.",
            }
        )
        messages.append({"role": "user", "content": clean_text})

        schema_instr = f"Respond only in JSON format: {json.dumps(self.json_schema)}"
        messages.append({"role": "user", "content": schema_instr})

        messages.append(
            {"role": "assistant", "content": "{"}
        )  # Start with '{' to hint JSON
        return messages

    def _reason(self, text: str) -> str:
        """
        Internal reasoning step to help the model better understand the text for summarization.
        """
        messages = [
            {
                "role": "user",
                "content": """
                    Read the following text and identify its main points, key arguments, and overall purpose.
                    Provide a brief, summarized analysis that will help in generating an accurate and concise summary.
                    """,
            },
            {
                "role": "user",
                "content": f"""
                    {text}
                    """,
            },
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )

        reason_summary = resp.choices[0].message.content.strip()
        return reason_summary

    def summarize(self, text: str) -> str:
        """
        Generates a summary for `text`.
        Optionally uses an internal reasoning step for better quality.
        """
        reason_summary = None
        if self.use_reason:
            reason_summary = self._reason(text)

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

        result = parsed.get("summary")
        # Validate that the result is a string
        if not isinstance(result, str):
            raise ValueError(
                f"Invalid response schema, expected a string for 'summary', got: {parsed}"
            )

        # dispatch and return, passing original_text
        self._dispatch(summary=result, original_text=text)
        return result
