# TextTools

![PyPI](https://img.shields.io/pypi/v/hamtaa-texttools)
![License](https://img.shields.io/pypi/l/hamtaa-texttools)

## 📌 Overview

**TextTools** is a high-level **NLP toolkit** built on top of **LLMs**.  

It provides both **sync (`TheTool`)** and **async (`AsyncTheTool`)** APIs for maximum flexibility.

It provides ready-to-use utilities for **translation, question detection, categorization, NER extraction, and more** - designed to help you integrate AI-powered text processing into your applications with minimal effort.

---

## ✨ Features

TextTools provides a rich collection of high-level NLP utilities,
Each tool is designed to work with structured outputs.

- **`categorize()`** - Classify text into given categories
- **`extract_keywords()`** - Extract keywords from the text
- **`extract_entities()`** - Perform Named Entity Recognition (NER)
- **`is_question()`** - Detect if the input is phrased as a question
- **`to_question()`** - Generate questions from the given text / subject
- **`merge_questions()`** - Merge multiple questions into one
- **`augment()`** - Rewrite text in different augmentations
- **`summarize()`** - Summarize the given text
- **`translate()`** - Translate text between languages
- **`propositionize()`** - Convert a text into atomic, independent, meaningful sentences 
- **`is_fact()`** - Check whether a statement is a fact based on the source text
- **`run_custom()`** - Custom tool that can do almost anything

---

## 🚀 Installation

Install the latest release via PyPI:

```bash
pip install -U hamtaa-texttools
```

---

## 📊 Tool Quality Tiers

| Status | Meaning | Tools | Safe for Production? |
|--------|---------|----------|-------------------|
| **✅ Production** | Evaluated, tested, stable. | `categorize()` (list mode), `extract_keywords()`, `extract_entities()`, `is_question()`, `to_question()`, `merge_questions()`, `augment()`, `summarize()`, `run_custom()` | **Yes** - ready for reliable use. |
| **🧪 Experimental** | Added to the package but **not fully evaluated**. | `categorize()` (tree mode), `translate()`, `propositionize()`, `is_fact()` | **Use with caution** |

---

## ⚙️ Additional Parameters

- **`with_analysis: bool`** → Adds a reasoning step before generating the final output.
**Note:** This doubles token usage per call.

- **`logprobs: bool`** → Returns token-level probabilities for the generated output. You can also specify `top_logprobs=<N>` to get the top N alternative tokens and their probabilities.  
**Note:** This feature works if it's supported by the model.

- **`output_lang: str`** → Forces the model to respond in a specific language.

- **`user_prompt: str`** → Allows you to inject a custom instruction into the model alongside the main template.

- **`temperature: float`** → Determines how creative the model should respond. Takes a float number between `0.0` and `2.0`.

- **`validator: Callable (Experimental)`** → Forces the tool to validate the output result based on your validator function. Validator should return a boolean. If the validator fails, TheTool will retry to get another output by modifying `temperature`. You can also specify `max_validation_retries=<N>`.

- **`priority: int (Experimental)`** → Affects processing order in queues.  
**Note:** This feature works if it's supported by the model and vLLM.

- **`timeout: float`** → Maximum time in seconds to wait for the response before raising a timeout error.  
**Note:** This feature is only available in `AsyncTheTool`.


---

## 🧩 ToolOutput

Every tool of `TextTools` returns a `ToolOutput` object which is a BaseModel with attributes:
- **`result: Any`**
- **`analysis: str`**
- **`logprobs: list`**
- **`errors: list[str]`**
- **`ToolOutputMetadata`**  
    - **`tool_name: str`**
    - **`processed_at: datetime`**
    - **`execution_time: float`**

**Note:** You can use `repr(ToolOutput)` to print your output with all the details.

---

## 🧨 Sync vs Async
| Tool         | Style   | Use case                                    |
|--------------|---------|---------------------------------------------|
| `TheTool`    | Sync    | Simple scripts, sequential workflows        |
| `AsyncTheTool` | Async | High-throughput apps, APIs, concurrent tasks |

---

## ⚡ Quick Start (Sync)

```python
from openai import OpenAI
from texttools import TheTool

client = OpenAI(base_url="your_url", API_KEY="your_api_key")
model = "model_name"

the_tool = TheTool(client=client, model=model)

detection = the_tool.is_question("Is this project open source?")
print(repr(detection))
```

---

## ⚡ Quick Start (Async)

```python
import asyncio
from openai import AsyncOpenAI
from texttools import AsyncTheTool

async def main():
    async_client = AsyncOpenAI(base_url="your_url", api_key="your_api_key")
    model = "model_name"

    async_the_tool = AsyncTheTool(client=async_client, model=model)
    
    translation_task = async_the_tool.translate("سلام، حالت چطوره؟", target_language="English")
    keywords_task = async_the_tool.extract_keywords("This open source project is great for processing large datasets!")

    (translation, keywords) = await asyncio.gather(translation_task, keywords_task)
    
    print(repr(translation))
    print(repr(keywords))

asyncio.run(main())
```

---

## 👍 Use Cases

Use **TextTools** when you need to:

- 🔍 **Classify** large datasets quickly without model training   
- 🧩 **Integrate** LLMs into production pipelines (structured outputs)  
- 📊 **Analyze** large text collections using embeddings and categorization  

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to **open issues, suggest new features, or submit pull requests**.  
