from typing import Any, Dict, List, Optional
import json
from openai import OpenAI
from texttools.base.base_keyword_extractor import BaseKeywordExtractor


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
        temperature: float = 0.0,
        prompt_template: str = None,
        handlers: List[Any] = None,
        **client_kwargs: Any,
    ):
        super().__init__(handlers)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs

        self.use_reason = use_reason
        self.prompt_template = prompt_template

        # Define the JSON schema for keyword extraction
        self.json_schema = {
            "keywords": ["string"]  # Represents an array of strings
        }

    def _build_messages(
        self, text: str, reason: Optional[str] = None
    ) -> List[Dict[str, str]]:
        clean_text = self.preprocess(text)

        messages: List[Dict[str, str]] = []

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
        schema_instr = f"Respond only in JSON format: {json.dumps(self.json_schema)}"
        messages.append({"role": "user", "content": schema_instr})

        messages.append(
            {"role": "assistant", "content": "{"}
        )  # Start with '{' to hint JSON
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

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )

        reason_summary = resp.choices[0].message.content.strip()
        return reason_summary

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extracts keywords from `text`.
        Optionally uses an internal reasoning step for better accuracy.
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

        result = parsed.get("keywords")
        # Validate that the result is a list of strings
        if not isinstance(result, list) or not all(
            isinstance(item, str) for item in result
        ):
            raise ValueError(
                f"Invalid response schema, expected a list of strings for 'keywords', got: {parsed}"
            )

        # dispatch and return
        self._dispatch({"original_text": text, "keywords": result})
        return result
