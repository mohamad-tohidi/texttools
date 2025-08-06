from typing import Any, Optional, Literal
import re
import json

from openai import OpenAI
from pydantic import BaseModel

from texttools.base.base_categorizer import BaseCategorizer
from texttools.formatter import Gemma3Formatter
from texttools.handlers import ResultHandler


class Output(BaseModel):
    reason: str
    main_tag: Literal[
        "باورهای دینی",
        "اخلاق اسلامی",
        "احکام و فقه",
        "تاریخ اسلام و شخصیت ها",
        "منابع دینی",
        "دین و جامعه/سیاست",
        "عرفان و معنویت",
        "هیچکدام",
    ] = None


class GemmaCategorizer(BaseCategorizer):
    """
    Categorizer for Gemma-style models. It requires a predefined Enum of categories
    to choose from and returns an Enum member.
    Outputs a single string: category name.

    Allows optional extra instructions via `prompt_template`.
    """

    CATEGORIES = [
        "باورهای دینی",
        "اخلاق اسلامی",
        "احکام و فقه",
        "تاریخ اسلام و شخصیت ها",
        "منابع دینی",
        "دین و جامعه/سیاست",
        "عرفان و معنویت",
        "هیچکدام",
    ]

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        chat_formatter: Optional[Any] = None,
        use_reason: bool = False,
        temperature: float = 0.0,
        prompt_template: Optional[str] = None,
        handlers: Optional[list[ResultHandler]] = None,
        **client_kwargs: Any,
    ):
        super().__init__(handlers=handlers)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs
        self.chat_formatter = chat_formatter or Gemma3Formatter()
        self.use_reason = use_reason
        self.prompt_template = prompt_template

    def _build_messages(
        self, text: str, reason: Optional[str] = None
    ) -> list[dict[str, str]]:
        """
        Builds the message list for the LLM API call for categorization.
        """
        clean_text = self.preprocess(text)

        messages: list[dict[str, str]] = []

        if self.prompt_template:
            messages.append({"role": "user", "content": self.prompt_template})

        if reason:
            messages.append(
                {"role": "user", "content": f"Based on this analysis: {reason}"}
            )

        main_prompt = f"""
        تو یک متخصص علوم دینی هستی
        من به عنوان کاربر یک متن به تو میدم و از تو میخوام که
        اون متن رو در یکی از دسته بندی های زیر طبقه بندی کنی
        {[category for category in self.CATEGORIES]}
        در خروجی که از تو خواسته شده بخشی با عنوان reason وجود دارد
        در اون بخش، دلیل انتخاب دسته بندی رو به صورت خلاصه بیاور
        در پاسخ، فقط یک شیء JSON برگردان که دو فیلد داشته باشد:

        {{
        "reason": "دلیل کوتاه برای انتخاب این دسته‌بندی",
        "main_tag": "نام دقیق یکی از دسته‌بندی‌ها"
        }}
        متنی که باید طبقه بندی کنی:
        """
        messages.append({"role": "user", "content": main_prompt})
        messages.append({"role": "user", "content": clean_text})

        restructured = self.chat_formatter.format(messages=messages)

        return restructured

    def _reason(self, text: str) -> str:
        """
        Internal reasoning step to help the model analyze the text for categorization.
        """
        reason_prompt = f"""
        هدف ما طبقه بندی متن هست
        متن رو بخون و ایده اصلی و آنالیزی کوتاه از اون رو ارائه بده
        بسیار خلاصه باشه خروجی تو
        نهایتا 20 کلمه 
        {text}
        """
        messages = [{"role": "user", "content": reason_prompt}]

        restrucruted = self.chat_formatter.format(messages=messages)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=restrucruted,
            temperature=self.temperature,
            **self.client_kwargs,
        )

        reason_summary = resp.choices[0].message.content.strip()
        return reason_summary

    def categorize(self, text: str):
        """
        Categorizes `text` by selecting an appropriate member from the predefined Enum.
        Optionally uses an internal reasoning step for better accuracy.
        """
        reason_summary = None
        if self.use_reason:
            reason_summary = self._reason(text)

        messages = self._build_messages(text, reason_summary)

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            # Use guided choice to have structured output
            extra_body={"guided_choice": self.CATEGORIES},
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "categorization",
                    "schema": Output.model_json_schema(),
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
        main_tag = entities["main_tag"]
        # Dispatch and return - Note: _dispatch expects dict
        self._dispatch(results={"main_tag": main_tag})
        return {"main_tag": main_tag}
