import os

from dotenv import load_dotenv
from openai import OpenAI

from texttools import TheTool

# Load environment variables from .env
load_dotenv()

# Load API key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")
model = "google/gemma-3n-e4b-it"

# Create OpenAI client
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

# Create an instance of TheTool
t = TheTool(client=client, model=model, output_lang="Persian")

# Rewriter
mode1 = t.rewrite(
    "چه کسی به عنوان اولین نفر وارد بهشت خواهد شد؟",
    mode="positive",
)["result"]
print(f"positive: {mode1}")

mode2 = t.rewrite(
    "چه کسی به عنوان اولین نفر وارد بهشت خواهد شد؟",
    mode="negative",
)["result"]
print(f"negative: {mode2}")

mode3 = t.rewrite(
    "چه کسی به عنوان اولین نفر وارد بهشت خواهد شد؟",
    mode="hard_negative",
)["result"]
print(f"hard_negative: {mode3}")
