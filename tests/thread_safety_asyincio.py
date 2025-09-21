import os
import asyncio

from dotenv import load_dotenv
from openai import OpenAI

from texttools import TheTool

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
tool = TheTool(client=client, model="gpt-4o-mini")

# Sample inputs
keyword_inputs = [
    "Tomorrow we will be dead by the car crash",
    "Python is fun",
    "The weather is nice",
]

question_inputs = [
    "Will it rain tomorrow?",
    "Is Python easy to learn?",
    "Do you like philosophy?",
]


async def run_keywords(text):
    result = await asyncio.to_thread(tool.extract_keywords, text, True)
    return text, result["result"]


async def run_question(text):
    result = await asyncio.to_thread(tool.detect_question, text)
    return text, result


async def main():
    tasks = []

    # Schedule both types of tasks
    for text in keyword_inputs * 5:  # repeat for stress test
        tasks.append(run_keywords(text))

    for text in question_inputs * 5:
        tasks.append(run_question(text))

    results = await asyncio.gather(*tasks)

    print(f"Completed {len(results)} tasks successfully\n")

    # Print sample outputs
    for text, output in results:
        print(f"{text} -> {output}")


if __name__ == "__main__":
    asyncio.run(main())
