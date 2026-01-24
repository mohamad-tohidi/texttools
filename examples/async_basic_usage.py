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

async_the_tool = AsyncTheTool(client=client, model=MODEL)


async def main():
    category_task = async_the_tool.categorize(
        "سلام حالت چطوره؟",
        categories=["هیچکدام", "دینی", "فلسفه"],
        timeout=4,
    )
    keywords_task = async_the_tool.extract_keywords(
        "Tomorrow, we will be dead by the car crash", mode="auto", timeout=3
    )
    entities_task = async_the_tool.extract_entities(
        "We will be dead by the car crash", timeout=1
    )
    detection_task = async_the_tool.is_question("We will be dead by the car crash")
    question_task = async_the_tool.to_question(
        "We will be dead by the car crash", mode="from_text", number_of_questions=2
    )
    merged_task = async_the_tool.merge_questions(
        ["چرا ما موجوداتی اجتماعی هستیم؟", "چرا باید در کنار هم زندگی کنیم؟"],
        mode="stepwise",
    )
    augmentations_task = async_the_tool.augment(
        "چرا ما انسان ها، موجوداتی اجتماعی هستیم؟",
        mode="positive",
    )
    summary_task = async_the_tool.summarize(
        "Tomorrow, we will be dead by the car crash"
    )
    translation_task = async_the_tool.translate(
        "سلام حالت چطوره؟", target_lang="English"
    )
    propositionize_task = async_the_tool.propositionize(
        "جنگ جهانی دوم در سال ۱۹۳۹ آغاز شد و آلمان به لهستان حمله کرد."
    )
    is_fact_task = async_the_tool.is_fact(
        text="امام نهم در ایران به خاک سپرده شد",
        source_text="حرم مطهر امام رضا علیه السلام در مشهد مقدس هست",
    )
    outputs = await asyncio.gather(
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

    for output in outputs:
        print(output.to_json())


if __name__ == "__main__":
    asyncio.run(main())
