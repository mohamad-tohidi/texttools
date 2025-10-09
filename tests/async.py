import os
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
import time

from texttools import AsyncTheTool

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
tool = AsyncTheTool(client=client, model="gpt-4o-mini")


async def main():
    s = time.time()
    original = "چه کسی به عنوان اولین نفر وارد بهشت خواهد شد؟"
    positive_task = tool.rewrite(original, mode="positive", with_analysis=True)
    negative_task = tool.rewrite(original, mode="negative", with_analysis=True)
    hard_negative_task = tool.rewrite(
        original, mode="hard_negative", with_analysis=True
    )
    translation_task = tool.translate(
        original, target_language="Enlgish", with_analysis=True
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
