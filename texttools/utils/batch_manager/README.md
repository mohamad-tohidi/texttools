# How to USE?

```python
from texttools import SimpleBatchManager


class IsQuestion(BaseModel):
    result: bool

client = OpenAI(api_key="your-api-key")

processor = SimpleBatchManager(
    client=client,
    model="gpt-4o-mini",
    prompt_template="You are a binary classifier. Answer only with `true` or `false`.",
    output_model=IsQuestion,
)

inputs = [
    "Is this a question?",
    "Tell me a story.",
    "What time is it?",
    "Run the code."
]

processor.start(inputs, job_name="detect_questions")

if processor.check_status("detect_questions") == "completed":
    result = processor.fetch_results("detect_questions")
    print(result["results"])

```

