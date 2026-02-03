import asyncio
import logging
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from texttools import BatchTheTool

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

# Set logger level
logging.basicConfig(level=logging.INFO)

# Initialize clients
client = AsyncOpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)
batch_the_tool = BatchTheTool(
    client=client, model=MODEL, raise_on_error=False, max_concurrency=3
)


async def main():
    print("=== Batch Categorization ===")
    categories = await batch_the_tool.categorize(
        texts=[
            "انسان‌ها به چه دلایلی هنر را خلق می‌کنند و چه تاثیری بر جامعه دارد؟",
            "اثرات تغییرات آب‌وهوایی بر کشاورزی مدرن",
            "بررسی سیستم‌های اقتصادی در جوامع پیشرفته",
            "مطالعه رفتار سلول‌های عصبی در حافظه",
        ],
        categories=["هیچکدام", "فلسفه", "علوم تجربی", "علوم اجتماعی"],
    )

    for i, result in enumerate(categories):
        print(f"\nText {i + 1} Result:")
        print(f"Category: {result.result}")

    print("\n=== Batch Keyword Extraction ===")
    keywords_list = await batch_the_tool.extract_keywords(
        texts=[
            "Climate change is causing unprecedented weather patterns, rising sea levels, and global ecological disruptions.",
            "Artificial intelligence is transforming industries through automation and data analysis.",
            "Renewable energy sources are becoming more cost-effective and widely adopted.",
            "Quantum computing promises to revolutionize cryptography and drug discovery.",
        ],
        mode="auto",
        output_lang="English",
    )

    for i, result in enumerate(keywords_list):
        print(f"\nText {i + 1} Keywords:")
        print(f"Keywords: {result.result}")


if __name__ == "__main__":
    asyncio.run(main())
