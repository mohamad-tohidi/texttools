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


preprocessed = translator.translate("ملاقات در دادگاه خاطراتی از مراودات رهبر معظم انقلاب و آیت‌الله طالقانی سخنان سید مهدی طالقانی، فرزند مرحوم آیت‌الله طالقانی در اوج روزهای فتنه که گفته بود: «پدرم از هر جریانی به غیر از جریان اصیل ولایت فقیه تبری می‌جست»، برای آن‌ها که قصد داشتند آن عالم وارسته را مصادره کنند، بسیار گران آمد. امروز، چهاردهم اسفندماه سال ۸۹، همایش نکوداشت این مجاهد نستوه و به تعبیر امام(ره) ابوذر زمان است. به همین مناسبت، خاطرات فرزند مرحوم طالقانی از مراودات پدرش و آیت‌الله خامنه‌ای خواندنی خواهد بود", "English", "Persian")
print(preprocessed)