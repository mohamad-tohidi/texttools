import os
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

from texttools import AsyncTheTool

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
tool = AsyncTheTool(client=client, model="gpt-4o-mini")


async def main():
    original = "چه کسی به عنوان اولین نفر وارد بهشت خواهد شد؟"
    result1 = await tool.rewrite(original, mode="positive")
    result2 = await tool.rewrite(original, mode="negative")
    result3 = await tool.rewrite(original, mode="hard_negative")

    print("Original:", original)
    print("Positive rewrite:", result1["result"])
    print("Negative rewrite:", result2["result"])
    print("Hard negative rewrite:", result3["result"])


if __name__ == "__main__":
    asyncio.run(main())
