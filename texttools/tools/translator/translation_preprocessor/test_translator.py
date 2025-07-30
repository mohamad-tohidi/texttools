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


preprocessed = translator.preprocess("دعای ندبه یادگاری از پایه گذار فقه جعفری در فراق قائم(عج) دعای ندبه، ازجمله ادعیه معروفی است که از امام ششم شیعیان، امام جعفر صادق(ع)، پایه گذار فقه جعفری در فراق امام زمان (عج) نقل شده است و مضامین بسیار قوی در این منبع غنی شیعی بیان شده است.")
print(preprocessed)