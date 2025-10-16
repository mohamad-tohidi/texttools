# TextTools

## 📌 Overview

**TextTools** is a high-level **NLP toolkit** built on top of modern **LLMs**.  

It provides both **sync (`TheTool`)** and **async (`AsyncTheTool`)** APIs for maximum flexibility.

It provides ready-to-use utilities for **translation, question detection, keyword extraction, categorization, NER extractor, and more** — designed to help you integrate AI-powered text processing into your applications with minimal effort.

**Thread Safety:** All methods in AsyncTheTool are thread-safe, allowing concurrent usage across multiple threads without conflicts.

---

## ✨ Features

TextTools provides a rich collection of high-level NLP utilities built on top of LLMs.  
Each tool is designed to work out-of-the-box with structured outputs (JSON / Pydantic).

- **`categorize()`** - Classifies text into Islamic studies categories 
- **`is_question()`** - Binary detection of whether input is a question
- **`extract_keywords()`** - Extracts keywords from text
- **`extract_entities()`** - Named Entity Recognition (NER) system
- **`summarize()`** - Text summarization
- **`text_to_question()`** - Generates questions from text
- **`merge_questions()`** - Merges multiple questions with different modes
- **`rewrite()`** - Rewrites text with different wording/meaning
- **`subject_to_question()`** - Generates questions about a specific subject
- **`translate()`** - Text translation between languages
- **`run_custom()`** - Allows users to define a custom tool with arbitrary BaseModel

---

## ⚙️ `with_analysis`, `logprobs`, `output_lang`, and `user_prompt` parameters

TextTools provides several optional flags to customize LLM behavior:

- **`with_analysis=True`** → Adds a reasoning step before generating the final output. Useful for debugging, improving prompts, or understanding model behavior.  
Note: This doubles token usage per call because it triggers an additional LLM request.

- **`logprobs=True`** → Returns token-level probabilities for the generated output. You can also specify `top_logprobs=<N>` to get the top N alternative tokens and their probabilities.  

- **`output_lang="en"`** → Forces the model to respond in a specific language. The model will ignore other instructions about language and respond strictly in the requested language.

- **`user_prompt="..."`** → Allows you to inject a custom instruction or prompt into the model alongside the main template. This gives you fine-grained control over how the model interprets or modifies the input text.

All these flags can be used individually or together to tailor the behavior of any tool in **TextTools**.

---

## 🚀 Installation

Install the latest release via PyPI:

```bash
pip install -U hamtaa-texttools
```

---

## Sync vs Async
| Tool         | Style   | Use case                                    |
|--------------|---------|---------------------------------------------|
| `TheTool`    | Sync    | Simple scripts, sequential workflows        |
| `AsyncTheTool` | Async | High-throughput apps, APIs, concurrent tasks |

---

## ⚡ Quick Start (Sync)

```python
from openai import OpenAI
from pydantic import BaseModel
from texttools import TheTool

# Create your OpenAI client
client = OpenAI(base_url = "your_url", API_KEY = "your_api_key")

# Specify the model
model = "gpt-4o-mini"

# Create an instance of TheTool
# Note: You can give parameters to TheTool so that you don't need to give them to each tool
the_tool = TheTool(client=client, model=model, with_analysis=True, output_lang="English")

# Example: Question Detection
detection = the_tool.is_question("Is this project open source?", logprobs=True, top_logprobs=2)
print(detection["result"])
print(detection["logprobs"])
# Output: True

# Example: Translation
# Note: You can overwrite with_analysis if defined at TheTool
print(the_tool.translate("سلام، حالت چطوره؟", target_language="English", with_analysis=False)["result"])
# Output: "Hi! How are you?"

# Example: Custom Tool
# Note: Output model should only contain result key
# Everything else will be ignored
class Custom(BaseModel):
  result: list[list[dict[str, int]]]

custom_prompt = "Something"
custom_result = the_tool.run_custom(custom_prompt, Custom)
print(custom_result)
```

---

## ⚡ Quick Start (Async)

```python
import asyncio
from openai import AsyncOpenAI
from texttools import AsyncTheTool

async def main():
    # Create your async OpenAI client
    async_client = AsyncOpenAI(base_url="your_url", api_key="your_api_key")

    # Specify the model
    model = "gpt-4o-mini"

    # Create an instance of AsyncTheTool
    the_tool = AsyncTheTool(client=async_client, model=model)

    # Example: Async Translation
    result = await the_tool.translate("سلام، حالت چطوره؟", target_language="English")
    print(result["result"])
    # Output: "Hi! How are you?"

asyncio.run(main())
```

---

## 📚 Use Cases

Use **TextTools** when you need to:

- 🔍 **Classify** large datasets quickly without model training  
- 🌍 **Translate** and process multilingual corpora with ease  
- 🧩 **Integrate** LLMs into production pipelines (structured outputs)  
- 📊 **Analyze** large text collections using embeddings and categorization  
- 👍 **Automate** common text-processing tasks without reinventing the wheel  

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to **open issues, suggest new features, or submit pull requests**.  

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
