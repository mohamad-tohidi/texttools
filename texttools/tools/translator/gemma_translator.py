import json
from typing import Any, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from texttools.base.base_translator import BaseTranslator
from texttools.formatter.gemma3_formatter import Gemma3Formatter


class PreprocessorOutput(BaseModel):
    """
    List of proper-name strings extracted from the source text.
    """

    entities: List[str] = Field(
        description="All proper names found in the text; return an empty list if none."
    )


class GemmaTranslator(BaseTranslator):
    """
    Translator for Gemma-style models using structured JSON prompts.
    Outputs only the translated text, without any additional structure.
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
        self.client: OpenAI = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs
        self.chat_formatter = chat_formatter or Gemma3Formatter()
        self.use_reason = use_reason
        self.prompt_template = prompt_template

    def _build_messages(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        reason: Optional[str] = None,
        proper_names: Optional[list[str]] = None,
    ) -> list[dict[str, str]]:
        """Constructs a single, comprehensive JSON prompt for the translation task."""

        prompt_data = {
            "role": "Expert Translator",
            "task": f"Translate the following text from {source_language or 'the original language'} to {target_language}.",
            "input_text": text,
            "rules": {
                "proper_names": {
                    "instruction": "These names MUST NOT be translated. Only transliterate them to preserve their phonetic form.",
                    "list": proper_names if proper_names else "None",
                }
            },
            "output_instructions": [
                "Provide ONLY the translated text.",
                "Do not include any explanations, comments, or markdown formatting.",
            ],
        }

        if reason:
            prompt_data["context"] = {
                "preliminary_analysis": reason,
                "instruction": "Use this analysis to inform the translation.",
            }

        # The entire set of instructions is formatted into a single JSON string
        content = json.dumps(prompt_data, ensure_ascii=False)
        messages = [{"role": "user", "content": content}]

        # Optional additional JSON template for more complex rules
        if self.prompt_template:
            messages.append({"role": "user", "content": self.prompt_template})

        return self.chat_formatter.format(messages=messages)

    def _reason(self, text: str, target_language: str) -> str:
        """Internal reasoning step using a JSON prompt to analyze text before translation."""

        prompt_data = {
            "task": "Analyze the provided text to identify potential translation challenges.",
            "analysis_points": [
                "Identify idioms or colloquialisms.",
                "Note any cultural references.",
                "Point out complex grammatical structures.",
                "List all proper nouns that should be transliterated, not translated.",
            ],
            "input_text": text,
            "output_instructions": {
                "language": target_language,
                "format": "A concise, bulleted list.",
                "important_rule": "DO NOT TRANSLATE the original text.",
                "length": "must be less than 200 words.",
            },
        }

        messages = [
            {
                "role": "user",
                "content": json.dumps(prompt_data, ensure_ascii=False),
            }
        ]

        restructured = self.chat_formatter.format(messages=messages)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=restructured,
            temperature=self.temperature,
            **self.client_kwargs,
        )

        return completion.choices[0].message.content.strip()

    def preprocess(self, text: str) -> PreprocessorOutput:
        """Preprocessor that finds proper names using a structured JSON prompt."""

        prompt_data = {
            "task_description": "Extract all proper names of people from the provided text.",
            "input_text": text,
            "output_format": {
                "schema": {"entities": ["string"]},
                "instruction": "Return a JSON object matching this schema. If no names are found, the 'entities' list must be empty.",
            },
        }

        messages = [
            {
                "role": "user",
                "content": json.dumps(prompt_data, ensure_ascii=False),
            }
        ]

        restructured = self.chat_formatter.format(messages=messages)

        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=restructured,
            response_format=PreprocessorOutput,
            temperature=self.temperature,
            extra_body={
                "guided_decoding_backend": "auto",
            },
            **self.client_kwargs,
        )

        return completion.choices[0].message.parsed

    def translate(
        self, text: str, target_language: str, source_language: Optional[str] = None
    ) -> str:
        """Translates text using a structured JSON-based workflow."""

        # 1. Preprocess: Extract proper names
        extracted_data = self.preprocess(text)
        proper_names = extracted_data.entities

        # 2. Reason (optional): Analyze the text for challenges
        reason_summary = None
        if self.use_reason:
            reason_summary = self._reason(text, target_language)

        # 3. Translate: Build the final prompt and get the translation
        messages = self._build_messages(
            text, target_language, source_language, reason_summary, proper_names
        )

        # For debugging purposes, let's see the final prompt
        print("--- Translation Request ---")
        print(f"Original: {text}")
        print(
            f"Translating to {target_language} from {source_language or 'original'}..."
        )
        if reason_summary:
            print(f"Reasoning Analysis:\n{reason_summary}")
        print("--- Final JSON Prompt Sent to Model ---")
        # Pretty-print the JSON content from the message
        print(json.dumps(json.loads(messages[0]["content"]), ensure_ascii=False))
        print("---------------------------")

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        response = completion.choices[0].message.content

        self._dispatch(
            {
                "original_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "translated_text": response,
            }
        )
        return response
