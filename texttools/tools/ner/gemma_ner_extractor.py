import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from texttools.base.base_ner_extractor import BaseNERExtractor


class GemmaNERExtractor(BaseNERExtractor):
    """
    Named Entity Recognition (NER) system for Gemma-style models with optional reasoning step.
    Outputs JSON with a single array field: {"entities": [{"text": "...", "type": "..."}, ...]}.

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
        # Handlers can be any type that implements a .handle method
        handlers: Optional[List[Any]] = None,
        **client_kwargs: Any,
    ):
        super().__init__(handlers)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs

        self.use_reason = use_reason
        self.prompt_template = prompt_template

        # Define the JSON schema for NER output
        # This specifies an array of objects, where each object has 'text' (string) and 'type' (string)
        self.json_schema = {
            "entities": [
                {
                    "text": "string",
                    "type": "string",
                }
            ]
        }

    def _build_messages(
        self, text: str, reason: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Builds the message list for the LLM API call for entity extraction.
        """
        clean_text = self.preprocess(text)

        messages: List[Dict[str, str]] = []

        if self.prompt_template:
            messages.append({"role": "user", "content": self.prompt_template})

        if reason:
            messages.append(
                {"role": "user", "content": f"Based on this analysis: {reason}"}
            )

        messages.append(
            {
                "role": "user",
                "content": "Identify and extract all named entities (e.g., PER, ORG, LOC, DAT, etc.) from the following text. For each entity, provide its text and a clear type. Respond as a JSON array of objects.",
            }
        )
        messages.append({"role": "user", "content": clean_text})

        # Ensure the schema is dumped as a valid JSON string for the LLM
        schema_instr = f"Respond only in JSON format: {json.dumps(self.json_schema)}"
        messages.append({"role": "user", "content": schema_instr})

        messages.append(
            {"role": "assistant", "content": "{"}
        )  # Hint to start JSON output
        return messages

    def _reason(self, text: str) -> str:
        """
        Internal reasoning step to help the model identify potential entities and their context.
        """
        messages = [
            {
                "role": "user",
                "content": """
                    Read the following text and identify any proper nouns, key concepts, or specific mentions that might represent named entities.
                    Provide a brief, summarized analysis that could help in categorizing these entities.
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

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extracts named entities from `text`.
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

        # Robustly parse JSON, even if the LLM adds extraneous text before the JSON
        if not raw.startswith("{"):
            raw = "{" + raw
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON for NER: {e}\nRaw output: {raw}")

        entities = parsed.get("entities")

        # Validate that 'entities' is a list and contains dictionaries with 'text' and 'type'
        if not isinstance(entities, list) or not all(
            isinstance(item, dict)
            and "text" in item
            and "type" in item
            and isinstance(item["text"], str)
            and isinstance(item["type"], str)
            for item in entities
        ):
            raise ValueError(
                f"Invalid response schema for NER. Expected 'entities' as a list of dicts with 'text' and 'type', got: {parsed}"
            )

        # dispatch and return
        self._dispatch(entities=entities, original_text=text)
        return entities
