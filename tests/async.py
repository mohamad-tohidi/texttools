import os
import asyncio

from dotenv import load_dotenv
from openai import OpenAI

from texttools import AsyncTheTool

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
tool = AsyncTheTool(client=client, model="gpt-4o-mini")


async def main():
    result = await tool.categorize("سلام، حالت چطوره؟")
    print(result["result"])


if __name__ == "__main__":
    asyncio.run(main())
