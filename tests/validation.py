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


# Define validator
def validate(result: Any) -> bool:
    return "هیچکدام" not in result


# Categorizer
category = t.categorize("سلام حالت چطوره؟", validator=validate)
print(category)
