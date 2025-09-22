# Batch

## Overview
`SimpleBatchManager` is a lightweight Python utility that simplifies working with the [OpenAI Batch API](https://platform.openai.com/docs/guides/batch). It allows you to:

- Efficiently submit **large volumes of prompts** for processing.  
- Automatically manage **job state** (saving, resuming, clearing).  
- Enforce **structured output** with [Pydantic](https://docs.pydantic.dev/).  
- Optionally define **custom JSON schemas** for advanced validation.  
- Fetch results and handle errors gracefully.  

This tool is especially useful if you need to process thousands of inputs (NER, classification, summarization, etc.) without manually managing batch job lifecycles.

---


## Usage

### 1. Import libraries

```python
from openai import OpenAI

from texttools import SimpleBatchManager
```

### 2. Define output model

```python
class MyOutput(BaseModel):
    is_question: bool
```

### 2. Initialize the Manager

```python
client = OpenAI()

prompt_template = "You are a binary classifier. Answer only with `true` or `false"

model = "gpt-4o-mini"

manager = SimpleBatchManager(
    client=client,
    model=model,
    prompt_template=prompt_template,
    output_model=MyOutput
)
```

### 3. Start a Batch Job

```python
inputs = [
    "Is this a question?",
    "Tell me a story.",
    "What time is it?",
    "Run the code."
]

manager.start(inputs, job_name="detect_questions")
```

### 4. Fetch Results

```python
if processor.check_status("detect_questions") == "completed":
    result = processor.fetch_results("detect_questions")
    print(result["results"])
```
