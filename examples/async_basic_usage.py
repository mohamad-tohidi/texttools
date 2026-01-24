import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from texttools import AsyncTheTool

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

# Initialize clients
client = AsyncOpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)
async_the_tool = AsyncTheTool(client=client, model=MODEL)


async def main():
    category_task = async_the_tool.categorize(
        "انسان‌ها به چه دلایلی فلسفه و هنر را توسعه داده‌اند و چه تاثیری بر جامعه دارند؟",
        categories=["هیچکدام", "فلسفه", "علوم تجربی", "علوم اجتماعی"],
        timeout=4,
    )
    keywords_task = async_the_tool.extract_keywords(
        "Global warming is leading to melting glaciers, rising sea levels, and extreme weather events worldwide.",
        mode="auto",
    )
    entities_task = async_the_tool.extract_entities(
        "Marie Curie discovered radium and polonium, pioneering research in radioactivity.",
    )
    is_question_task = async_the_tool.is_question(
        "What are the consequences of deforestation on biodiversity?"
    )
    questions_task = async_the_tool.to_question(
        "Renewable energy sources like solar, wind, and hydro are crucial to reduce carbon emissions and combat climate change.",
        mode="from_text",
        number_of_questions=2,
    )
    merged_question_task = async_the_tool.merge_questions(
        [
            "چرا انسان‌ها در طول تاریخ تمدن ساخته‌اند؟",
            "چه عواملی باعث پیشرفت علمی و فرهنگی جوامع شده‌اند؟",
        ],
        mode="stepwise",
    )
    augmented_task = async_the_tool.augment(
        "انسان‌ها به دلیل کنجکاوی و تعامل اجتماعی، علم و هنر را توسعه داده‌اند.",
        mode="positive",
    )
    summary_task = async_the_tool.summarize(
        "The research paper analyzes the psychological and social effects of remote work during prolonged periods of isolation."
    )
    translation_task = async_the_tool.translate(
        "علم و تکنولوژی می‌تواند کیفیت زندگی انسان‌ها را به شکل چشمگیری ارتقا دهد.",
        target_lang="English",
    )
    propositions_task = async_the_tool.propositionize(
        "Alexander Fleming discovered penicillin in 1928, revolutionizing the treatment of bacterial infections."
    )
    is_fact_task = async_the_tool.is_fact(
        text="The Eiffel Tower is located in Paris, France.",
        source_text="The Eiffel Tower, an iron lattice tower, is situated on the Champ de Mars in Paris, France.",
    )

    outputs = await asyncio.gather(
        category_task,
        keywords_task,
        entities_task,
        is_question_task,
        questions_task,
        merged_question_task,
        augmented_task,
        summary_task,
        translation_task,
        propositions_task,
        is_fact_task,
    )

    for output in outputs:
        print(output.to_json())


if __name__ == "__main__":
    asyncio.run(main())
