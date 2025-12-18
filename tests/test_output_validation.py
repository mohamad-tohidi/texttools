import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from texttools import TheTool

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

client = OpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)

t = TheTool(client=client, model=MODEL)


# Define text to question validator
def validate(result: Any) -> bool:
    return "چیست؟" not in result


question = t.text_to_question(
    "زندگی",
    output_lang="Persian",
    validator=validate,
    max_validation_retries=0,
    temperature=1.0,
)
print(repr(question))
