import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from texttools import AsyncTheTool

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

client = AsyncOpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)

t = AsyncTheTool(client=client, model=MODEL)


async def main():
    category_task = t.categorize(
        "سلام حالت چطوره؟",
        categories=["هیچکدام", "دینی", "فلسفه"],
        timeout=0.5,
    )
    keywords_task = t.extract_keywords(
        "Tomorrow, we will be dead by the car crash", mode="auto"
    )
    entities_task = t.extract_entities(
        "We will be dead by the car crash", entities=["EVENT"]
    )
    detection_task = t.is_question("We will be dead by the car crash")
    question_task = t.to_question(
        "We will be dead by the car crash", mode="from_text", number_of_questions=2
    )
    merged_task = t.merge_questions(
        ["چرا ما موجوداتی اجتماعی هستیم؟", "چرا باید در کنار هم زندگی کنیم؟"],
        mode="stepwise",
        with_analysis=True,
        timeout=5.8,
    )
    augmentations_task = t.augment(
        "چرا ما انسان ها، موجوداتی اجتماعی هستیم؟",
        mode="positive",
        user_prompt="Be carefull",
    )
    summary_task = t.summarize("Tomorrow, we will be dead by the car crash")
    translation_task = t.translate("سلام حالت چطوره؟", target_language="English")
    propositionize_task = t.propositionize(
        "جنگ جهانی دوم در سال ۱۹۳۹ آغاز شد و آلمان به لهستان حمله کرد.",
        output_lang="Persian",
    )
    is_fact_task = t.is_fact(
        text="امام نهم در ایران به خاک سپرده شد",
        source_text="حرم مطهر امام رضا علیه السلام در مشهد مقدس هست",
    )
    results = await asyncio.gather(
        category_task,
        keywords_task,
        entities_task,
        detection_task,
        question_task,
        merged_task,
        augmentations_task,
        summary_task,
        translation_task,
        propositionize_task,
        is_fact_task,
    )

    for tool_output in results:
        print(repr(tool_output))


if __name__ == "__main__":
    asyncio.run(main())
