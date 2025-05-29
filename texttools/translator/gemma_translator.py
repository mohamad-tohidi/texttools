from typing import Any, Dict, List, Optional
import json
from openai import OpenAI
from texttools.base.base_translator import BaseTranslator


class GemmaTranslator(BaseTranslator):
    """
    Translator for Gemma-style models with optional reasoning step.
    Outputs JSON with a single string field: {"translated_text": "..."}.

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

        # Corrected: Use string "string" instead of Python type 'str'
        self.json_schema = {"translated_text": "string"}

    def _build_messages(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        clean_text = self.preprocess(text)

        messages: List[Dict[str, str]] = []

        if reason:
            messages.append(
                {"role": "user", "content": f"Based on this reasoning: {reason}"}
            )
            
        if source_language:
            messages.append(
                {
                    "role": "user",
                    "content": f"""
        Please translate the following text **from {source_language} to {target_language}**.

        Important instructions:
        - **Do NOT translate** proper nouns such as:
        - Names of people
        - Names of Qur’an surahs or religious terms
        - Place names
        - Other culturally specific or significant terms
        - Accurately preserve the original **meaning**, **tone**, and **context**.
        - Use fluent, natural language in the target translation.

        
        the target language is {target_language}
        

        """,
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": f"""
        Please translate the following text  to {target_language}**.

        Important instructions:
        - **Do NOT translate** proper nouns such as:
        - Names of people
        - Names of Qur’an surahs or religious terms
        - Place names
        - Other culturally specific or significant terms
        - Accurately preserve the original **meaning**, **tone**, and **context**.
        - Use fluent, natural language in the target translation.

        
        the target language is {target_language}
        

        """,
                }
            )

        schema_instr = f"Respond only in JSON format: {json.dumps(self.json_schema)}"
        messages.append({"role": "user", "content": schema_instr})

        if self.prompt_template:
            messages.append({"role": "user", "content": self.prompt_template})

        messages.append({"role": "user", "content": "here is the text that has to be translated:" + clean_text})
        messages.append({"role": "assistant", "content": "{"})
        return messages

    def _reason(
        self, text: str, target_language: str, source_language: Optional[str] = None
    ) -> str:
        """
        Internal reasoning step to help the model with translation.
        """
        messages = [
            {
                "role": "user",
                "content": f"""
                    We need to translate the following text.

                    First, carefully analyze the text for:
                    - Its content, context, tone, and any nuances that may affect translation from {source_language or "the original language"} to {target_language}.
                    - **Special terms that must be preserved or handled carefully**, such as:
                    - Names of people
                    - Names of Qur’an surahs or other religious references
                    - Place names
                    - Culturally significant terms or expressions

                    Identify and **remember** these special words or phrases for accurate handling during translation.  
                    Provide a concise summary of your analysis, highlighting:
                    - The key ideas or themes
                    - Any challenging parts of the text
                    - A list (or description) of special terms and how they should be treated in translation
                    
                    DONT OVER ANALYZE
                    
                    
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

    def translate(
        self, text: str, target_language: str, source_language: Optional[str] = None
    ) -> str:
        """
        Translates `text` from `source_language` (if provided) to the `target_language`.
        Optionally uses an internal reasoning step for better accuracy.
        """
        reason_summary = None
        if self.use_reason:
            reason_summary = self._reason(text, target_language, source_language)

        messages = self._build_messages(
            text, target_language, source_language, reason_summary
        )
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
            try:
                prefix, sep, remainder = raw.partition('"translated_text":')
                value_str = remainder.strip()
                if value_str.startswith('"') and value_str.endswith('"}'):
                    inner = value_str[1:-2]
                    sanitized_inner = inner.replace('"', '\\"')
                    raw = f"{prefix}{sep}\"{sanitized_inner}\"}}"
                    parsed = json.loads(raw)
                else:
                    raise
            except Exception:
                raise ValueError(f"Failed to parse JSON after sanitation: {e}\nRaw output: {raw}")

        result = parsed.get("translated_text")
        if not isinstance(result, str):  # This check is still valid and important!
            raise ValueError(f"Invalid response schema, got: {parsed}")

        # dispatch and return
        self._dispatch(
            {
                "original_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "translated_text": result,
            }
        )
        return result
