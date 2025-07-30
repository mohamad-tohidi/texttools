from openai import OpenAI

from gemma_translator_beta import GemmaTranslator

API_KEY = "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"
model = "google/gemma-3n-e4b-it:free"

client = OpenAI(
	api_key=API_KEY,
 	base_url="https://openrouter.ai/api/v1"
)

translator = GemmaTranslator(
    client = client,
    model=model
)


preprocessed = translator.preprocess("حقیقت رحمانی بودن یا شیطانی بودن «عرفان حلقه» را باید از آسیب دیدگانی پرسید که به واسطه فرادرمانی نزد محمدعلی طاهری و مربیان وی رفته و هم اکنون در مراکز مغز و اعصاب در شهرهای مختلف کشور بستری هستند.")
print(preprocessed)