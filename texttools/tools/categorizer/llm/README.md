```python
from enum import Enum
from openai import OpenAI

# 1️⃣ Define your categories
class ProductCategory(Enum):
    CAR    = "car"
    PHONE  = "phone"
    LAPTOP = "laptop"

# 2️⃣ Instantiate the OpenAI client
client = OpenAI()

# 3️⃣ Create your categorizer
cat = LLMCategorizer(
    client=client,
    categories=ProductCategory,
    model="gpt-4o-2024-08-06",
    temperature=0.0,
    max_tokens=10,             # any other kwargs flow through
    prompt_template=(
        "Classify the following user query into one of these products:"
    )
)

# 4️⃣ Run it
result_enum = cat.categorize("How do I change a car tire?")
print(result_enum)         # e.g. ProductCategory.CAR
print(result_enum.value)   # "car"
```

here is an example of how to use it