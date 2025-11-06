import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from texttools import TheTool

# Load environment variables from .env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

# Create OpenAI client
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Create an instance of TheTool
t = TheTool(client=client, model=MODEL)


# Define categorizer validator
def validate1(result: Any) -> bool:
    return "هیچکدام" not in result


# Define text to question validator
def validate2(result: Any) -> bool:
    return "زندگی" not in result


# Categorizer
category = t.categorize("سلام حالت چطوره؟", validator=validate1)
print(category)

print("-" * 40)

# Question from Text Generator
question = t.text_to_question("زندگی", validator=validate2, output_lang="Persian")
print(question)
