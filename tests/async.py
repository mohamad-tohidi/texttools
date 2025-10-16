import os
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
import time

from texttools import AsyncTheTool

# Load environment variables from .env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

model = "gpt-4o-mini"

# Create AsyncOpenAI client
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# Create an instance of TheTool
tool = AsyncTheTool(client=client, model=model)


async def main():
    s = time.time()
    original = "چه کسی به عنوان اولین نفر وارد بهشت خواهد شد؟"
    positive_task = tool.rewrite(original, mode="positive", with_analysis=True)
    negative_task = tool.rewrite(original, mode="negative", with_analysis=True)
    hard_negative_task = tool.rewrite(
        original, mode="hard_negative", with_analysis=True
    )
    translation_task = tool.translate(
        original, target_language="English", with_analysis=True
    )

    positive, negative, hard_negative, translation = await asyncio.gather(
        positive_task, negative_task, hard_negative_task, translation_task
    )
    e = time.time()

    print("Original:", original)
    print("Positive:", positive["result"])
    print("Negative:", negative["result"])
    print("Hard negative:", hard_negative["result"])
    print("Translation:", translation["result"])
    print(e - s)


if __name__ == "__main__":
    asyncio.run(main())
