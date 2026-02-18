# TextTools

![PyPI](https://img.shields.io/pypi/v/hamtaa-texttools)
![License](https://img.shields.io/pypi/l/hamtaa-texttools)

## 📌 Overview

**TextTools** is a high-level **NLP toolkit** built on top of **LLMs**.  

It provides three API styles for maximum flexibility:
- Sync API (`TheTool`) - Simple, sequential operations
- Async API (`AsyncTheTool`) - High-performance async operations
- Batch API (`BatchTheTool`) - Process multiple texts in parallel with built-in concurrency control

It provides ready-to-use utilities for **translation, question detection, categorization, NER extraction, and more** - designed to help you integrate AI-powered text processing into your applications with minimal effort.

---

## ✨ Features

TextTools provides a collection of high-level NLP utilities.
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
| **✅ Production** | Evaluated and tested. | `categorize()`, `extract_keywords()`, `extract_entities()`, `is_question()`, `to_question()`, `merge_questions()`, `augment()`, `summarize()`, `run_custom()` | **Yes** - ready for reliable use. |
| **🧪 Experimental** | Added to the package but **not fully evaluated**. |  `translate()`, `propositionize()`, `is_fact()` | **Use with caution** |

---

## ⚙️ Additional Parameters

- **`with_analysis: bool`** → Adds a reasoning step before generating the final output.
**Note:** This doubles token usage per call.

- **`logprobs: bool`** → Returns token-level probabilities for the generated output. You can also specify `top_logprobs=<N>` to get the top N alternative tokens and their probabilities.  
**Note:** This feature works if it's supported by the model.

- **`output_lang: str`** → Forces the model to respond in a specific language.

- **`user_prompt: str`** → Allows you to inject a custom instruction into the model alongside the main template.

- **`temperature: float`** → Determines how creative the model should respond. Takes a float number between `0.0` and `2.0`.

- **`normalize: bool`** → Whether to apply text cleaning (removing separator lines and normalizing quotation marks) before sending to the LLM.

- **`validator: Callable (Experimental)`** → Forces the tool to validate the output result based on your validator function. Validator should return a boolean. If the validator fails, TheTool will retry to get another output by modifying `temperature`. You can also specify `max_validation_retries=<N>`.

- **`priority: int (Experimental)`** → Affects processing order in queues.  
**Note:** This feature works if it's supported by the model and vLLM.

- **`timeout: float`** → Maximum time in seconds to wait for the response before raising a timeout error.  
**Note:** This feature is only available in `AsyncTheTool`.

- **`raise_on_error: bool`** → (`TheTool/AsyncTheTool`) Raise errors (True) or return them in output (False). Default is True.

- **`max_concurrency: int`** → (`BatchTheTool` only) Maximum number of concurrent API calls. Default is 5.

---

## 🧩 ToolOutput

Every tool of `TextTools` returns a `ToolOutput` object which is a BaseModel with attributes:

- **`result: Any`**
- **`analysis: str`**
- **`logprobs: list`**
- **`errors: list[str]`**
- **`ToolOutputMetadata`**
    - **`tool_name: str`**
    - **`processed_by: str`**
    - **`processed_at: datetime`**
    - **`execution_time: float`**
    - **`token_usage: TokenUsage`**
        - **`completion_usage: CompletionUsage`**
            - **`prompt_tokens: int`**
            - **`completion_tokens: int`**
            - **`total_tokens: int`**
        - **`analyze_usage: AnalyzeUsage`**
            - **`prompt_tokens: int`**
            - **`completion_tokens: int`**
            - **`total_tokens: int`**
        - **`total_tokens: int`**

- Serialize output to JSON using the `model_dump_json()` method.
- Verify operation success with the `is_successful()` method.
- Convert output to a dictionary with the `model_dump()` method.

**Note:** For BatchTheTool: Each method returns a `list[ToolOutput]` containing results for all input texts.

---

## 🧨 Sync vs Async vs Batch
| Tool | Style | Use Case | Best For |
|------|-------|----------|----------|
| `TheTool` | **Sync** | Simple scripts, sequential workflows | • Quick prototyping<br>• Simple scripts<br>• Sequential processing<br>• Debugging |
| `AsyncTheTool` | **Async** | High-throughput applications, APIs, concurrent tasks | • Web APIs<br>• Concurrent operations<br>• High-performance apps<br>• Real-time processing |
| `BatchTheTool` | **Batch** | Process multiple texts efficiently with controlled concurrency | • Bulk processing<br>• Large datasets<br>• Parallel execution<br>• Resource optimization |

---

## ⚡ Quick Start (Sync)

```python
from openai import OpenAI
from texttools import TheTool

client = OpenAI(base_url="your_url", API_KEY="your_api_key")
model = "model_name"

the_tool = TheTool(client=client, model=model)

detection = the_tool.is_question("Is this project open source?")
print(detection.model_dump_json())
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
    
    print(translation.model_dump_json())
    print(keywords.model_dump_json())

asyncio.run(main())
```

## ⚡ Quick Start (Batch)

```python
import asyncio
from openai import AsyncOpenAI
from texttools import BatchTheTool

async def main():
    async_client = AsyncOpenAI(base_url="your_url", api_key="your_api_key")
    model = "model_name"
    
    batch_the_tool = BatchTheTool(client=async_client, model=model, max_concurrency=3)
    
    categories = await batch_tool.categorize(
        texts=[
            "Climate change impacts on agriculture",
            "Artificial intelligence in healthcare",
            "Economic effects of remote work",
            "Advancements in quantum computing",
        ],
        categories=["Science", "Technology", "Economics", "Environment"],
    )
    
    for i, result in enumerate(categories):
        print(f"Text {i+1}: {result.result}")

asyncio.run(main())
```

---

## ✅ Use Cases

Use **TextTools** when you need to:

- 🔍 **Classify** large datasets quickly without model training   
- 🧩 **Integrate** LLMs into production pipelines (structured outputs)  
- 📊 **Analyze** large text collections using embeddings and categorization

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions from the community! - see the [CONTRIBUTING](CONTRIBUTING.md) file for details.

## 📚 Documentation

For detailed documentation, architecture overview, and implementation details, please visit the [docs](docs) directory.