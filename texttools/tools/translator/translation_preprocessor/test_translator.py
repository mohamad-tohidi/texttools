from openai import OpenAI

from gemma_translator import GemmaTranslator

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

ner = GemmaNERExtractor(
    client=client,
    model=model
)

translator.translate("سند روایت «...لولا فاطمه لما خلقتکما» چیست و معنای این روایت را توضیح دهید.", "English", "Persian")