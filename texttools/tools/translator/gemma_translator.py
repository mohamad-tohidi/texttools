from typing import Any, Dict, List, Optional
from openai import OpenAI
from texttools.base.base_translator import BaseTranslator
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
        messages.append(
            {
                "role": "system",
                "content": (
                    f"""You are a {source_language}-to-{target_language} translator.
                    Output only and only the translated text without any explanations or additions."""
                ),
            }
        )

        if reason:
            messages.append(
                {
                    "role": "user",
                    "content": f"""Based on the analysis conducted, translate the following text {"from" + source_language if source_language else ""} to {target_language}.
                    The text to be translated is: "{clean_text}"
                    The analysis conducted: {reason}""",
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": f"""Translate the following text from {source_language or "original"} to {target_language}:
                    {clean_text}""",
                }
            )

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
        messages = [
            {
                "role": "system",
                "content": f"""Analyze the following text and identify important linguistic considerations for translation.
                               Do not translate the text. Point out any idioms, cultural references, or complex structures that need special attention.
                               Also, list all proper nouns that should not be translated. Write your analysis in the {target_language}.""",
            },
            {"role": "user", "content": text},
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
