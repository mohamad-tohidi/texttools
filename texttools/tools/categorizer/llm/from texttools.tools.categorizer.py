from gemma_categorizer import GemmaCategorizer
from openai import OpenAI

API_KEY = "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"
model = "google/gemma-3n-e4b-it:free"

client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

categorizer = GemmaCategorizer(client=client, model=model)

categorizer.categorize("حضرت محمد و مسلم بن عقیل بسیار افراد خوبی بودند")
