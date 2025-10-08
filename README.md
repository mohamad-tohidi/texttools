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

- **Categorizer** → Zero-finetuning text categorization for fast, scalable classification.  
- **Keyword Extractor** → Identify the most important keywords in a text.  
- **Question Merger** → Merge the provided questions, preserving all the main points 
- **NER (Named Entity Recognition) Extractor** → Extract people, places, organizations, and other entities.  
- **Question Detector** → Determine whether a text is a question or not.  
- **Question Generator From Text** → Generate high-quality, context-relevant questions from provided text.
- **Question Generator From Subject** → Generate high-quality, context-relevant questions from a subject.
- **Rewriter** → Rewrite text while preserving meaning or without it.
- **Summarizer** → Condense long passages into clear, structured summaries. 
- **Translator** → Translate text across multiple languages, with support for custom rules.
- **Custom Tool** → Allows users to define a custom tool with arbitrary BaseModel. 

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
pip install -U hamta-texttools
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
detection = the_tool.detect_question("Is this project open source?", logpobs=True, top_logprobs=2)
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
custom_result = the_tool.custom_tool(custom_prompt, Custom)
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
