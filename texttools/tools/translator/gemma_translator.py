from typing import Any, Optional
import re
import json

from openai import OpenAI
from pydantic import BaseModel

from texttools.base.base_translator import BaseTranslator
from texttools.formatter.gemma3_fromatter import Gemma3Formatter


# Pydantic BaseModel to specify the output format of preprocessor
# Preprocessor's job is to extract proper names
class PreprocessorOutput(BaseModel):
    text: str
    text_type: str


class GemmaTranslator(BaseTranslator):
    """
    Translator for Gemma-style models with optional reasoning step.
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
        self.client = client
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
        messages: list[dict[str, str]] = []

        # This prompt will enforce LLM to output in the required format
        # This prompt also gives initial information about translation like languages and proper names
        enforce_prompt = f"""
        You are a {source_language}-to-{target_language} translator.
        Important Rule: The following are proper names and must NOT be translated.
        They must be only transliterated into {target_language}.
        That means preserving their phonetic form without changing their meaning.
        Apply the rule for **ALL** of following proper names.
        Proper names (do not translate **** of them):
        {proper_names if proper_names else "None"}
        If any proper name is found in the text, you MUST only transliterate it.
        Output only the translated text. No comments, no explanations, no markdown.
        """
        messages.append({"role": "user", "content": enforce_prompt})

        clean_text = text.strip()
        if reason:
            reason_prompt = f"""Based on the analysis conducted, translate the following text {"from" + source_language if source_language else ""} to {target_language}.
            The text to be translated is: "{clean_text}"
            The analysis conducted: {reason}"""
            messages.append({"role": "user", "content": reason_prompt})
        else:
            regular_prompt = f"""Translate the following text from {source_language or "original"} to {target_language}: 
            {clean_text}"""
            messages.append({"role": "user", "content": regular_prompt})

        # Optional additional template
        if self.prompt_template:
            messages.append({"role": "user", "content": self.prompt_template})

        return messages

    def _reason(
        self, text: str, target_language: str, source_language: Optional[str] = None
    ) -> str:
        """
        Internal reasoning step to help the model with translation.
        """

        reason_step_prompt = f"""Analyze the following text and identify important linguistic considerations for translation.
        Do not translate the text. Point out any idioms, cultural references, or complex structures that need special attention.
        Also, list all proper nouns that should not be translated. Write your analysis in the {target_language}."""
        messages = [
            {"role": "user", "content": reason_step_prompt},
            {"role": "user", "content": text},
        ]

        restructured = self.chat_formatter.format(messages=messages)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=restructured,
            temperature=self.temperature,
            **self.client_kwargs,
        )

        return completion.choices[0].message.content.strip()

    def translate(
        self, text: str, target_language: str, source_language: Optional[str] = None
    ) -> str:
        """
        Translates text and returns only the translated string.
        """

        # Extract proper names to tell the LLM what names not to translate, but to transliterate
        extracted = self.preprocess(text)
        proper_names = [e["text"] for e in extracted]

        reason_summary = None
        if self.use_reason:
            reason_summary = self._reason(text, target_language, source_language)

        messages = self._build_messages(
            text, target_language, source_language, reason_summary, proper_names
        )
        print(f"Original: {text}")
        print(
            f"Translating to {target_language} from {source_language or 'original'}..."
        )
        print(
            f"Reasoning: {reason_summary}" if reason_summary else "No reasoning used."
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        response = completion.choices[0].message.content.strip()

        self._dispatch(
            {
                "original_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "translated_text": response,
            }
        )
        return response

    def preprocess(self, text: str) -> list:
        """Preprocessor that finds proper names of Islamic figures. The extractions will be given to the
        LLm in order to know that it shouldn't translate them, but transliterate them."""

        messages: list[dict[str, str]] = []

        main_prompt = """You must detect proper names of people.
        Your task is to extract a JSON list of entities from the given input. For each entity, include:
        text: The exact matched string from the original.
        type: Only include "Proper Name" for actual names of real people. 
        If there is no proper name in the following text, return empty json."""

        messages.append({"role": "user", "content": main_prompt})

        text_prompt = f"""The text to be extracted is:{text}"""
        messages.append({"role": "user", "content": text_prompt})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "NER",
                    "schema": PreprocessorOutput.model_json_schema(),
                },
            },
            temperature=self.temperature,
            **self.client_kwargs,
        )
        response = completion.choices[0].message.content

        # Remove Markdown-style triple backticks and any optional language tag like "json"
        if response.startswith("```"):
            response = re.sub(r"^```(?:json)?\s*|```$", "", response.strip())

        entities = json.loads(response)

        return entities
