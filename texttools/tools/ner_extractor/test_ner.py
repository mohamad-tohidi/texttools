from openai import OpenAI

from texttools.tools.ner_extractor.ner_extractor import NERExtractor

client = OpenAI(
    base_url="http://185.208.182.157:9310/v1", api_key="hamta_T0k3n879875412"
)

tool = NERExtractor(client=client, model="gemma-3", use_reason=False)
c = tool.extract_entities(
    "در ساختمانی که در شهر قم، در کشور ایران، توسط عباس ساخته شد، دوستان زیادی کار کردند."
)
print(c)
