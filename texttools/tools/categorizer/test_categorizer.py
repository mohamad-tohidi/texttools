from openai import OpenAI

from categorizer import Categorizer

client = OpenAI(
    base_url="http://185.208.182.157:9310/v1", api_key="hamta_T0k3n879875412"
)

tool = Categorizer(client=client, model="gemma-3", use_reason=False)
c = tool.categorize("یک متن نمونه برای بررسی")
print(c)
