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
    ) -> List[Dict[str, str]]:
        clean_text = self.preprocess(text)
        messages: List[Dict[str, str]] = []

        # Enforce pure translation output
        enforce_prompt = f"""You are a {source_language}-to-{target_language} translator.
        Output only and only the translated text without any explanations or additions."""
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
        reason_summary = None
        if self.use_reason:
            reason_summary = self._reason(text, target_language, source_language)

        messages = self._build_messages(
            text, target_language, source_language, reason_summary
        )
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
        print(f"Translated: {translated}")
        return translated
    
    def preprocess(self, text) -> str:
        """Preprocessor that tags protected elements (e.g., hadiths, Quran and proper names)"""
        
        # Create the message for tagging
        messages: List[Dict[str, str]] = []

        main_prompt = """
        You're an expert in identifying only *verified* Quranic verses (in Arabic) and *authentic* hadiths from canonical Islamic collections. You must also detect well-known Islamic proper names.

        Your job is to **return a JSON list of extracted entities** from the given input, where each item includes:
        - `text`: The exact matched string from the original.
        - `type`: One of these strictly limited values:
        - `"QURAN/HADITH"` – **only** for Arabic Quran verses or Arabic verifiable hadiths from major Sunni collections (e.g., Sahih Bukhari, Muslim). Do not include Persian religious content, general moral teachings, or commentary.
        - `"Proper Name"` – for proper names of Islamic figures (e.g., "بلعم باعورا", "علی", "بلال حبشی").

        Strict Exclusion Rules:
        - Do **NOT** include interpretations, religious opinion, or moral lessons—even if they *sound* Islamic.
        - Do **NOT** include any Persian or non-Arabic statements.
        - Do **NOT** guess hadiths unless their text is a **verifiable match** to known hadith literature.

        Absolutely exclude any religious-sounding content that is NOT directly from the Quran in Arabic or authentic hadithin Arabic sources.
        If the text is in Persian, or an interpretation, or simply *discusses religion*, it must NOT be labeled as QURAN/HADITH.
        If the text is not in Arabic script or does not resemble verse structure, do NOT classify it as Quran/Hadith.
        """

        messages.append({"role": "user", "content": main_prompt})
        
        # Append the text
        text_prompt = f"""The text is:{text}"""
        messages.append({"role": "user", "content": text_prompt})

        # Enforce json output
        json_schema = {
            "entities": [
                {
                    "text": "نَحْنُ أَقْرَبُ إِلَیْهِ مِنْ حَبْلِ الْوَرید",
                    "type": "QURAN/HADITH",
                },
                {
                    "text": "بلعم باعورا",
                    "type": "Proper Name"
                }
            ]
        }
        enforce_prompt = f"""Respond only in JSON format: {json.dumps(json_schema)}
        No addition, no extra things"""
        messages.append({"role": "user", "content": enforce_prompt})

        # Hint to start JSON output
        messages.append({"role": "assistant", "content": "{"})

        # Get the response via chat completion
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        extractions = completion.choices[0].message.content.strip()

        return extractions        
