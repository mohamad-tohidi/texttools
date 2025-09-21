import os

from dotenv import load_dotenv
from openai import OpenAI

from texttools import TheTool

# Load environment variables from .env
load_dotenv()

# Load API key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")
model = "gpt-4o-mini"

# Create OpenAI client
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

# Create an instance of TheTool
t = TheTool(client=client, model=model)

# Keyword Extractor
print("\n\nKEYWORD EXTRACTOR\n")
keywords = t.extract_keywords(
    "Tomorrow, we will be dead by the car crash", logprobs=True
)
logprobs = keywords["logprobs"]
for d in logprobs:
    print(d)
    print("-" * 40)

# Question Detector
print("\n\nQUESTION DETECTOR\n")
detection = t.detect_question(
    "What is the capital of France?", logprobs=True, top_logprobs=3
)
logprobs = detection["logprobs"]
for d in logprobs:
    print(d)
    print("-" * 40)
