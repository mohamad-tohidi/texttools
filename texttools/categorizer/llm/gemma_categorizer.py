from typing import Any, Dict, List, Optional
import json
from enum import Enum
from openai import OpenAI
from texttools.base.base_categorizer import BaseCategorizer

from texttools.handlers import ResultHandler


class GemmaCategorizer(BaseCategorizer):
    """
    Categorizer for Gemma-style models. It requires a predefined Enum of categories
    to choose from and returns an Enum member.
    Outputs JSON with a single string field: {"category": "..."}.

    Allows optional extra instructions via `prompt_template`.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        categories: Enum,  # REQUIRED: An Enum class representing categories
        use_reason: bool = False,
        temperature: float = 0.0,
        prompt_template: Optional[str] = None,
        handlers: Optional[List[ResultHandler]] = None,
        **client_kwargs: Any,
    ):
        # BaseCategorizer expects the Enum class directly
        super().__init__(categories=categories, handlers=handlers)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs

        # Extract actual string values/names from the Enum for prompting the LLM
        # We'll use the .name of the enum members
        self._category_names = [member.name for member in self.categories]

        self.use_reason = use_reason
        self.prompt_template = prompt_template

        self.json_schema = {
            "category": "string"  # LLM still returns a string, we convert it to Enum
        }

    def _build_messages(
        self, text: str, reason: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Builds the message list for the LLM API call for categorization.
        """
        clean_text = self.preprocess(text)

        messages: List[Dict[str, str]] = []

        if self.prompt_template:
            messages.append({"role": "user", "content": self.prompt_template})

        if reason:
            messages.append(
                {"role": "user", "content": f"Based on this analysis: {reason}"}
            )

        # Dynamically build the prompt with the allowed category names from the Enum
        category_list_str = ", ".join(
            [f"'{cat_name}'" for cat_name in self._category_names]
        )
        messages.append(
            {
                "role": "user",
                "content": f"Analyze the following text and determine its single most relevant category from this predefined list: {category_list_str}. You MUST choose one category name from this list. If the text does not fit any category perfectly, choose the one that is the closest match.",
            }
        )
        messages.append({"role": "user", "content": clean_text})

        schema_instr = f"Respond only in JSON format: {json.dumps(self.json_schema)}"
        messages.append({"role": "user", "content": schema_instr})

        messages.append(
            {"role": "assistant", "content": "{"}
        )  # Hint to start JSON output
        return messages

    def _reason(self, text: str) -> str:
        """
        Internal reasoning step to help the model analyze the text for categorization.
        """
        messages = [
            {
                "role": "user",
                "content": """
                    Read the following text and identify its core subject matter, key themes, and overall purpose.
                    Provide a brief, summarized analysis that could help in classifying its primary category.
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

    def categorize(self, text: str) -> Enum:
        """
        Categorizes `text` by selecting an appropriate member from the predefined Enum.
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
            raise ValueError(
                f"Failed to parse JSON for categorization: {e}\nRaw output: {raw}"
            )

        category_name = parsed.get("category")

        # --- Crucial step: Convert string output to Enum member ---
        if not isinstance(category_name, str):
            raise ValueError(
                f"Invalid response schema for categorization. Expected 'category' as a string, got: {parsed}"
            )

        try:
            # Attempt to convert the LLM's string output to an Enum member
            detected_category_enum = self.categories[category_name]
        except KeyError:
            raise ValueError(
                f"LLM returned category '{category_name}' which is not a valid member of the provided Enum: {self._category_names}. Raw output: {raw}"
            )

        # dispatch and return - Note: _dispatch expects Dict[str, Enum]
        self._dispatch(results={"category": detected_category_enum})
        return detected_category_enum
