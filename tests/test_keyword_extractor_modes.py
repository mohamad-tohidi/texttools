import os

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

# Auto mode
keywords = t.extract_keywords(
    "چرا در قرآن پاره ای از آیات تکرار شده است، و همچنین چرا بعضی از داستانها در موارد متعدّد آورده شده است؟!",
    output_lang="Farsi",
    mode="auto",
)
print(repr(keywords))

# Threshold mode
keywords = t.extract_keywords(
    "چرا در قرآن پاره ای از آیات تکرار شده است، و همچنین چرا بعضی از داستانها در موارد متعدّد آورده شده است؟!",
    output_lang="Farsi",
    mode="threshold",
)
print(repr(keywords))

# Count mode
keywords = t.extract_keywords(
    "چرا در قرآن پاره ای از آیات تکرار شده است، و همچنین چرا بعضی از داستانها در موارد متعدّد آورده شده است؟!",
    output_lang="Farsi",
    mode="count",
    number_of_keywords=2,
)
print(repr(keywords))
