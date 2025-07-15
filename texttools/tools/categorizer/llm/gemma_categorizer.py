from typing import Any, Dict, List, Literal, Optional

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
    Outputs JSON with a single string field: {"category": "..."}.

    Allows optional extra instructions via `prompt_template`.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        output_structure: BaseModel = Output,
        chat_formatter: Optional[Any] = None,
        use_reason: bool = False,
        temperature: float = 0.0,
        prompt_template: Optional[str] = None,
        handlers: Optional[List[ResultHandler]] = None,
        **client_kwargs: Any,
    ):
        super().__init__(handlers=handlers)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs
        self.output_structure = output_structure
        self.chat_formatter = chat_formatter or Gemma3Formatter()

        self.use_reason = use_reason
        self.prompt_template = prompt_template

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

        # # Dynamically build the prompt with the allowed category names from the Enum
        # category_list_str = ", ".join(
        #     [f"'{cat_name}'" for cat_name in self._category_names]
        # )
        messages.append(
            {
                "role": "user",
                "content": """
                تو یک متخصص علوم دینی هستی
                من به عنوان کاربر یک متن به تو میدم و از تو میخوام که
                اون متن رو در یکی از دسته بندی های زیر طبقه بندی کنی
                
        "باورهای دینی",
        "اخلاق اسلامی",
        "احکام و فقه",
        "تاریخ اسلام و شخصیت ها",
        "منابع دینی",
        "دین و جامعه/سیاست",
        "عرفان و معنویت",
        "هیچکدام",

                
                در خروجی که از تو خواسته شده بخشی با عنوان reason وجود دارد
                در اون بخش، دلیل انتخاب دسته بندی رو به صورت خلاصه بیاور


                متنی که باید طبقه بندی کنی:
                
                
                
                """,
            }
        )
        messages.append({"role": "user", "content": clean_text})

        # messages.append(
        #     {"role": "assistant", "content": "{"}
        # )  # Hint to start JSON output

        # this line will restructure the messages
        # based on the formatter that we provided
        # some models will require custom settings
        restructured = self.chat_formatter.format(messages=messages)

        return restructured

    def _reason(self, text: str) -> str:
        """
        Internal reasoning step to help the model analyze the text for categorization.
        """
        messages = [
            {
                "role": "user",
                "content": """
                هدف ما طبقه بندی متن هست
                متن رو بخون و ایده اصلی و آنالیزی کوتاه از اون رو ارائه بده
                
                بسیار خلاصه باشه خروجی تو
                
                    """,
            },
            {
                "role": "user",
                "content": f"""
                    {text}
                    """,
            },
        ]

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
            response_format=Output,
            temperature=self.temperature,
            extra_body=dict(guided_decoding_backend="auto"),
            **self.client_kwargs,
        )
        message = completion.choices[0].message

        category_name = message.parsed.main_tag

        # dispatch and return - Note: _dispatch expects dict
        self._dispatch(results={"main_tag": category_name})
        return {"main_tag": category_name}
