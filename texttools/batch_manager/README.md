# SimpleBatchManager

`SimpleBatchManager` is a lightweight Python utility that simplifies working with the [OpenAI Batch API](https://platform.openai.com/docs/guides/batch). It allows you to:

- Efficiently submit **large volumes of prompts** for processing.  
- Automatically manage **job state** (saving, resuming, clearing).  
- Enforce **structured output** with [Pydantic](https://docs.pydantic.dev/).  
- Optionally define **custom JSON schemas** for advanced validation.  
- Fetch results and handle errors gracefully.  

This tool is especially useful if you need to process thousands of inputs (NER, classification, summarization, etc.) without manually managing batch job lifecycles.

---


## 🚀 Usage

### 1. Define Your Output Model

```python
from pydantic import BaseModel

class MyOutput(BaseModel):
    label: str
    score: float
```

### 2. Initialize the Manager

```python
from openai import OpenAI
from simple_batch_manager import SimpleBatchManager

client = OpenAI()

prompt_template = "Classify the following text into sentiment categories."

manager = SimpleBatchManager(
    client=client,
    model="gpt-4.1-mini",
    output_model=MyOutput,
    prompt_template=prompt_template,
)
```

### 3. Start a Batch Job

```python
payload = [
    "I love this product!",
    "This was the worst experience ever.",
    "It’s okay, nothing special."
]

manager.start(payload, job_name="sentiment_job")
```

### 4. Check Job Status

```python
status = manager.check_status("sentiment_job")
print("Job status:", status)
```

### 5. Fetch Results

```python
results, log = manager.fetch_results("sentiment_job")

print("Results:", results)
print("Errors:", log)
```
