import os
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

from texttools import AsyncTheTool

# Load environment variables from .env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

# Create AsyncOpenAI client
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# Create an instance of TheTool
t = AsyncTheTool(client=client, model=MODEL)


async def main():
    category_task = t.categorize(
        "سلام حالت چطوره؟", categories=["هیچکدام", "دینی", "فلسفه"]
    )
    keywords_task = t.extract_keywords("Tomorrow, we will be dead by the car crash")
    entities_task = t.extract_entities("We will be dead by the car crash")
    detection_task = t.is_question("We will be dead by the car crash")
    question_task = t.text_to_question("We will be dead by the car crash")
    merged_task = t.merge_questions(
        ["چرا ما موجوداتی اجتماعی هستیم؟", "چرا باید در کنار هم زندگی کنیم؟"],
        mode="default",
        with_analysis=True,
    )
    rewritten_task = t.rewrite(
        "چرا ما انسان ها، موجوداتی اجتماعی هستیم؟",
        mode="positive",
        user_prompt="Be carefull",
    )
    questions_task = t.subject_to_question("Friendship", 3)
    summary_task = t.summarize("Tomorrow, we will be dead by the car crash")
    translation_task = t.translate("سلام حالت چطوره؟", target_language="English")
    propositionize_task = t.propositionize("جنگ جهانی دوم در سال ۱۹۳۹ آغاز شد و آلمان به لهستان حمله کرد.", output_lang="Persian")
    fact_check_task = t.fact_check(text="امام نهم در ایران به خاک سپرده شد", source_text="حرم مطهر امام رضا علیه السلام در مشهد مقدس هست")
    (
        category,
        keywords,
        entities,
        detection,
        question,
        merged,
        rewritten,
        questions,
        summary,
        translation,
        propositionize,
        fact_check,
    ) = await asyncio.gather(
        category_task,
        keywords_task,
        entities_task,
        detection_task,
        question_task,
        merged_task,
        rewritten_task,
        questions_task,
        summary_task,
        translation_task,
        propositionize_task,
        fact_check_task,
    )

    for tool_output in (
        category,
        keywords,
        entities,
        detection,
        question,
        merged,
        rewritten,
        questions,
        summary,
        translation,
        propositionize,
        fact_check,
    ):
        print(repr(tool_output))


if __name__ == "__main__":
    asyncio.run(main())
