import os
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

from texttools import AsyncTheTool

# Load environment variables from .env
load_dotenv()

# Load API key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")
model = "google/gemma-3n-e4b-it"

# Create OpenAI client
client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

# Create an instance of TheTool
t = AsyncTheTool(client=client, model=model)


async def main():
    original = "چرا خداوند نظام هستی را به طور کلی کمی با حساب و کتاب سبک تر برای ما طراحی نفرموده است؟"

    print(f"original: {original}")

    pos_task = t.rewrite(original, mode="positive", output_lang="Persian")
    neg_task = t.rewrite(original, mode="negative", output_lang="Persian")
    hard_neg_task = t.rewrite(original, mode="hard_negative", output_lang="Persian")
    pos, neg, hard_neg = await asyncio.gather(pos_task, neg_task, hard_neg_task)

    print(f"Pos: {pos['result']}")
    print(f"Neg: {neg['result']}")
    print(f"Hard Neg: {hard_neg['result']}")


if __name__ == "__main__":
    asyncio.run(main())
