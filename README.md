# TextTools

## 📌 Overview

**TextTools** is a high-level **NLP toolkit** built on top of modern **LLMs**.  

It provides both **sync (`TheTool`)** and **async (`AsyncTheTool`)** APIs for maximum flexibility.

It provides ready-to-use utilities for **translation, question detection, keyword extraction, categorization, NER extraction, and more** - designed to help you integrate AI-powered text processing into your applications with minimal effort.

---

## ✨ Features

TextTools provides a rich collection of high-level NLP utilities,
Each tool is designed to work with structured outputs (JSON / Pydantic).

- **`categorize()`** - Classifies text into given categories (You have to create a category tree)
- **`extract_keywords()`** - Extracts keywords from text
- **`extract_entities()`** - Named Entity Recognition (NER) system
- **`is_question()`** - Binary detection of whether input is a question
- **`text_to_question()`** - Generates questions from text
- **`merge_questions()`** - Merges multiple questions with different modes
- **`rewrite()`** - Rewrites text with different wording/meaning
- **`subject_to_question()`** - Generates questions about a specific subject
- **`summarize()`** - Text summarization
- **`translate()`** - Text translation between languages
- **`propositionize()`** - Convert text to atomic independence meaningful sentences 
- **`check_fact()`** - Check a statement is relevant to source text or not 
- **`run_custom()`** - Allows users to define a custom tool with an arbitrary BaseModel

---

## 📊 Tool Quality Tiers

| Status | Meaning | Use in Production? |
|--------|---------|-------------------|
| **✅ Production** | Evaluated, tested, stable. | **Yes** - ready for reliable use. |
| **🧪 Experimental** | Added to the package but **not fully evaluated**. Functional, but quality may vary. | **Use with caution** - outputs not yet validated. |

### Current Status
**Production Tools:**
- `categorize()` (list mode)
- `extract_keywords()`
- `extract_entities()`
- `is_question()`
- `text_to_question()`
- `merge_questions()`
- `rewrite()`
- `subject_to_question()`
- `summarize()`
- `run_custom()` (fine in most cases)

**Experimental Tools:**
- `categorize()` (tree mode)
- `translate()`
- `propositionize()`
- `check_fact()`
- `run_custom()` (not evaluated in all scenarios)

---

## ⚙️ `with_analysis`, `logprobs`, `output_lang`, `user_prompt`, `temperature`, `validator` and `priority` parameters

TextTools provides several optional flags to customize LLM behavior:

- **`with_analysis (bool)`** → Adds a reasoning step before generating the final output.
**Note:** This doubles token usage per call because it triggers an additional LLM request.

- **`logprobs (bool)`** → Returns token-level probabilities for the generated output. You can also specify `top_logprobs=<N>` to get the top N alternative tokens and their probabilities.  
**Note:** This feature works if it's supported by the model.

- **`output_lang (str)`** → Forces the model to respond in a specific language. The model will ignore other instructions about language and respond strictly in the requested language.

- **`user_prompt (str)`** → Allows you to inject a custom instruction or prompt into the model alongside the main template. This gives you fine-grained control over how the model interprets or modifies the input text.

- **`temperature (float)`** → Determines how creative the model should respond. Takes a float number from `0.0` to `2.0`.

- **`validator (Callable)`** → Forces TheTool to validate the output result based on your custom validator. Validator should return a bool (True if there were no problem, False if the validation fails.) If the validator fails, TheTool will retry to get another output by modifying `temperature`. You can specify `max_validation_retries=<N>` to change the number of retries.

- **`priority (int)`** → Task execution priority level. Higher values = higher priority. Affects processing order in queues.
**Note:** This feature works if it's supported by the model and vLLM.

**Note:** There might be some tools that don't support some of the parameters above.

---

## 🧩 ToolOutput

Every tool of `TextTools` returns a `ToolOutput` object which is a BaseModel with attributes:
- **`result: Any`** → The output of LLM
- **`analysis: str`** → The reasoning step before generating the final output
- **`logprobs: list`** → Token-level probabilities for the generated output 
- **`process: str`** → The tool name which processed the input
- **`processed_at: datetime`** → The process time
- **`execution_time: float`** → The execution time (seconds)
- **`errors: list[str]`** → Any error that have occured during calling LLM

**Note:** You can use `repr(ToolOutput)` to see details of your ToolOutput.

---

## 🚀 Installation

Install the latest release via PyPI:

```bash
pip install -U hamtaa-texttools
```

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

# Create your OpenAI client
client = OpenAI(base_url = "your_url", API_KEY = "your_api_key")

# Specify the model
model = "gpt-4o-mini"

# Create an instance of TheTool
the_tool = TheTool(client=client, model=model)

# Example: Question Detection
detection = the_tool.is_question("Is this project open source?", logprobs=True, top_logprobs=2)
print(detection.result)
print(detection.logprobs)
# Output: True + logprobs

# Example: Translation
translation = the_tool.translate("سلام، حالت چطوره؟" target_language="English", with_analysis=True)
print(translation.result)
print(translation.analysis)
# Output: "Hi! How are you?"  + analysis
```

---

## ⚡ Quick Start (Async)

```python
import asyncio
from openai import AsyncOpenAI
from texttools import AsyncTheTool

async def main():
    # Create your AsyncOpenAI client
    async_client = AsyncOpenAI(base_url="your_url", api_key="your_api_key")

    # Specify the model
    model = "gpt-4o-mini"

    # Create an instance of AsyncTheTool
    async_the_tool = AsyncTheTool(client=async_client, model=model)
    
    # Example: Async Translation and Keyword Extraction
    translation_task = async_the_tool.translate("سلام، حالت چطوره؟", target_language="English")
    keywords_task = async_the_tool.extract_keywords("Tomorrow, we will be dead by the car crash")

    (translation, keywords) = await asyncio.gather(translation_task, keywords_task)
    print(translation.result)
    print(keywords.result)

asyncio.run(main())
```

---

## 👍 Use Cases

Use **TextTools** when you need to:

- 🔍 **Classify** large datasets quickly without model training  
- 🌍 **Translate** and process multilingual corpora with ease  
- 🧩 **Integrate** LLMs into production pipelines (structured outputs)  
- 📊 **Analyze** large text collections using embeddings and categorization  

---

## 🔍 Logging

TextTools uses Python's standard `logging` module. The library's default logger level is `WARNING`, so if you want to modify it, follow instructions:


```python
import logging

# Default: warnings and errors only
logging.basicConfig(level=logging.WARNING)

# Debug everything (verbose)
logging.basicConfig(level=logging.DEBUG)

# Complete silence
logging.basicConfig(level=logging.CRITICAL)
```

---

## 📚 Batch Processing

Process large datasets efficiently using OpenAI's batch API.

## ⚡ Quick Start (Batch)

```python
from pydantic import BaseModel
from texttools import BatchJobRunner, BatchConfig

# Configure your batch job
config = BatchConfig(
    system_prompt="Extract entities from the text",
    job_name="entity_extraction",
    input_data_path="data.json",
    output_data_filename="results.json",
    model="gpt-4o-mini"
)

# Define your output schema
class Output(BaseModel):
    entities: list[str]

# Run the batch job
runner = BatchJobRunner(config, output_model=Output)
runner.run()
```

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to **open issues, suggest new features, or submit pull requests**.  

---

## 🌿 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
