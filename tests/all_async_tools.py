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
tool = AsyncTheTool(client=client, model=MODEL)


async def main():
    original = "چه کسی به عنوان اولین نفر وارد بهشت خواهد شد؟"
    positive_task = tool.rewrite(original, mode="positive")
    negative_task = tool.rewrite(original, mode="negative")
    hard_negative_task = tool.rewrite(original, mode="hard_negative")

    (
        positive,
        negative,
        hard_negative,
    ) = await asyncio.gather(positive_task, negative_task, hard_negative_task)

    print("Original:", original)
    print("Positive:", positive.result)
    print("Negative:", negative.result)
    print("Hard negative:", hard_negative.result)

    print(positive.errors)


if __name__ == "__main__":
    asyncio.run(main())
