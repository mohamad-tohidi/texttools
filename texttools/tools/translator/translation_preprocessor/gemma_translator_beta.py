from typing import Any, Dict, List, Optional

from openai import OpenAI
import json

from base_translator import BaseTranslator
from texttools.formatter.gemma3_fromatter import Gemma3Formatter

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
        handlers: List[Any] = None,
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
        proper_names: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        clean_text = text.strip()
        messages: List[Dict[str, str]] = []

        # Enforce pure translation output
        enforce_prompt = f"""You are a {source_language}-to-{target_language} translator.
        Output only and only the translated text without any explanations or additions.

        - Do NOT translate the following proper names: {proper_names if proper_names else 'None'}
        - For each of these proper names, transliterate them in {target_language}.

        DO NOT explain your output. Only return the translated text."""
        messages.append({"role": "user", "content": enforce_prompt})

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

        # The actual text
        # messages.append({"role": "user", "content": clean_text})

        # messages = self.chat_formatter.format(messages=messages)
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
            {"role": "user", "content": text}
        ]

        restructured = self.chat_formatter.format(messages=messages)
        completion = self.client.chat.completions.create(
            model=self.model,
            response_format=None,
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

        extracted = self.preprocess(text)
        proper_names = [e["text"] for e in extracted]

        reason_summary = None
        if self.use_reason:
            reason_summary = self._reason(text, target_language, source_language)

        messages = self._build_messages(text, target_language, source_language, reason_summary, proper_names)
        print()
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
        translated = completion.choices[0].message.content.strip()

        self._dispatch(
            {
                "original_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "translated_text": translated,
            }
        )
        #print(f"Translated: {translated}")
        return translated
    
    def preprocess(self, text) -> List:
        """Preprocessor that tags protected elements (e.g., hadiths, Quran and proper names)"""
        
        # Create the message for tagging
        messages: List[Dict[str, str]] = []

        main_prompt = """
        You must detect Islamic proper names of people ONLY.

        Your task is to extract a JSON list of entities from the given input. For each entity, include:
        - `text`: The exact matched string from the original.
        - `type`: Only include "Proper Name" for actual Islamic **names of real people**. 

        DO NOT include:
        - General or ambiguous descriptions without an actual name.
        - Roles, or adjectives of people.

        If there is no proper name in the following text, return empty json.
        """

        messages.append({"role": "user", "content": main_prompt})
        
        # Append the text
        text_prompt = f"""The text to be extracted is:{text}"""
        messages.append({"role": "user", "content": text_prompt})

        # Enforce json output
        json_schema = {
            "entities": [
                {
                    "text": "a proper name",
                    "type": "Proper Name"
                }
            ]
        }
        enforce_prompt = f"""Respond only in JSON format: {json.dumps(json_schema)} No additions."""
        messages.append({"role": "user", "content": enforce_prompt})

        # Hint to start JSON output
        messages.append({"role": "assistant", "content": "{"})

        # Get the response via chat completion
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        raw = response.choices[0].message.content.strip()

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

        return entities        
