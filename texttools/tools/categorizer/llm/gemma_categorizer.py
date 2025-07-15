import json
from enum import Enum
from typing import Any, Dict, List, Optional

from openai import OpenAI

from texttools.base.base_categorizer import BaseCategorizer
from texttools.formatter import Gemma3Formatter
from texttools.handlers import ResultHandler


class Category(Enum):
    CATEGORY_A = "باورهای دینی"
    CATEGORY_B = "اخلاق اسلامی"
    CATEGORY_C = "احکام و فقه"
    CATEGORY_D = "تاریخ اسلام و شخصیت‌ها"
    CATEGORY_E = "منابع دینی"
    CATEGORY_F = "دین و جامعه/سیاست"
    CATEGORY_G = "عرفان و معنویت"


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
        categories: Enum = Category,  # REQUIRED: An Enum class representing categories
        chat_formatter: Optional[Any] = None,
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

        self.chat_formatter = chat_formatter or Gemma3Formatter()
        # Extract actual string values/names from the Enum for prompting the LLM
        # We'll use the .name of the enum members
        self._category_names = [member.name for member in self.categories]

        self.use_reason = use_reason
        self.prompt_template = prompt_template

        self.json_schema = {
            "main_tag": "string",
            "secondary_tags": ["string"],
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

        # # Dynamically build the prompt with the allowed category names from the Enum
        # category_list_str = ", ".join(
        #     [f"'{cat_name}'" for cat_name in self._category_names]
        # )
        messages.append(
            {
                "role": "user",
                "content": """
تو یک متخصص علوم دینی اسلامی هستی و وظیفه‌ات دسته‌بندی سوالات دینی در یکی از هفت دسته زیر است. وظیفه دارید هر سوال دینی را با توجه به معنای دقیق آن در یکی از دسته‌های اصلی موضوعی قرار دهید، و در صورت نیاز، تگ‌های فرعی نیز به آن اختصاص دهید.

خروجی شما باید شامل دو بخش باشد:
main_tag (اجباری، فقط یکی از دسته‌های زیر)
secondary_tags (اختیاری، صفر تا چند مورد از دسته‌های دیگر)

در تعیین main_tag، توجه داشته باشید که نوع سوال (یعنی نوع پاسخ مورد انتظار) مهم‌تر از موضوع ظاهری آن است.

اگر سوال از جواز، حرمت، وجوب یا حکم شرعی یک عمل سوال کند، حتی اگر موضوع آن رفتاری اخلاقی یا ظاهری باشد (مانند اصلاح صورت، نگاه، یا لباس)، باید در دسته "احکام و فقه" قرار گیرد.

فقط اگر سوال به شکل کلی درباره خوبی یا بدی اخلاقی، نیت‌ها، صفات درونی یا رفتاری بدون اشاره به حکم شرعی باشد، در دسته "اخلاق اسلامی" قرار می‌گیرد.

تمام تگ‌ها باید دقیقاً از میان 7 دسته زیر انتخاب شوند:
"باورهای دینی"
پرسش‌هایی درباره اصول بنیادین دین اسلام مانند توحید، نبوت، معاد، امامت، خداشناسی، فلسفه دین، چیستی انسان، فطرت، عدل الهی، و همچنین دیدگاه اسلام درباره سایر ادیان، ادیان الهی یا غیرالهی، و باورهای فلسفی.
"اخلاق اسلامی"
سوال‌هایی که درباره فضائل و رذائل اخلاقی (شخصیتی، اجتماعی، روانی یا ارتباط با خدا) هستند، مانند پرسش درباره گناه، دعا، رفتار فردی یا اجتماعی بر اساس اخلاق اسلامی.
"احکام و فقه"
پرسش‌هایی درباره احکام عملی اسلام شامل عبادات (نماز، روزه، حج)، معاملات و امور مالی (خمس، زکات، ربا، بیع)، احکام فردی (طهارت، نجاست) و اجتماعی (حجاب، امر به معروف)، طبقه‌بندی شرعی اعمال (واجب، حرام...)، و فلسفه احکام.
"تاریخ اسلام و شخصیت‌ها"
پرسش‌هایی درباره وقایع تاریخی مهم در اسلام، زندگی پیامبران، اهل بیت، صحابه، خلفا، دشمنان دین، سیره و سبک زندگی ایشان، وقایعی مانند غدیر، عاشورا، هجرت، و جنبش‌ها و جریان‌های دینی در تاریخ اسلام.
"منابع دینی"
سوال‌هایی که مرتبط با قرآن، حدیث، تفسیر قرآن، معنای یک حدیث، اعتبار منابع دینی یا علوم قرآن و حدیث هستند.
"دین و جامعه/سیاست"
سوال‌هایی که درباره نقش دین در اجتماع و سیاست هستند، مانند مذاهب اسلامی، مراسم‌های دینی، مراکز دینی، حضور دین در عرصه‌های اجتماعی، یا مسائل سیاسی مرتبط با دین.
"عرفان و معنویت"
سلوک عرفانی، تهذیب نفس، ذکر، عشق الهی، حالات عرفانی، فناء و بقاء.

 قوانین مهم:
main_tag باید فقط یکی از دسته‌ها باشد.
secondary_tags یکی از دسته های بالا که به شکل فرعی یا ضمنی در سوال حضور دارند.
اگر سوال فقط در یک دسته جا می‌گیرد، secondary_tags را خالی بگذارید.
اگر سوال به هیچ‌کدام مرتبط نیست، main_tag را نامربوط قرار دهید و secondary_tags را خالی بگذارید.
 خروجی خود را دقیقاً به صورت زیر ارائه دهید:
json
{
  "main_tag": "احکام و فقه",
  "secondary_tags": ["اخلاق اسلامی", "عرفان و معنویت"]
}
""",
            }
        )
        messages.append({"role": "user", "content": clean_text})

        schema_instr = f"Respond only in JSON format: {json.dumps(self.json_schema)}"
        messages.append({"role": "user", "content": schema_instr})

        messages.append(
            {"role": "assistant", "content": "{"}
        )  # Hint to start JSON output

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

        restrucruted = self.chat_formatter.format(messages=messages)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=restrucruted,
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

        category_name = parsed.get("main_tag")

        # --- Crucial step: Convert string output to Enum member ---
        if not isinstance(category_name, str):
            raise ValueError(
                f"Invalid response schema for categorization. Expected 'category' as a string, got: {parsed}"
            )

        if not category_name not in self._category_names:
            raise ValueError(
                f"LLM returned category '{category_name}' which is not a valid member of the provided Enum: {self._category_names}. Raw output: {raw}"
            )

        secondary_tags = parsed.get("secondary_tags")
        if not isinstance(secondary_tags, list):
            raise ValueError(
                f"Invalid response schema for categorization. Expected 'secondary_tags' as a list, got: {parsed}"
            )
        if not all(isinstance(tag, str) for tag in secondary_tags):
            raise ValueError(
                f"Invalid response schema for categorization. All items in 'secondary_tags' should be strings, got: {parsed}"
            )

        if not all(tag in self._category_names for tag in secondary_tags):
            raise ValueError(
                f"LLM returned secondary tags that are not valid members of the provided Enum: {secondary_tags}. Valid categories: {self._category_names}"
            )

        # dispatch and return - Note: _dispatch expects dict
        self._dispatch(
            results={"main_tag": category_name, "secondary_tags": secondary_tags}
        )
        return {"main_tag": category_name, "secondary_tags": secondary_tags}
