# Text Tools

## 📌 Overview

**TextTools** is a high-level **NLP toolkit** built on top of modern **LLMs**.  
It provides ready-to-use utilities for **translation, question detection, keyword extraction, categorization, NER extractor, and more** — designed to help you integrate AI-powered text processing into your applications with minimal effort.

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

---

## 🔍 `with_analysis` Mode

The `with_analysis=True` flag enhances the tool's output by providing a detailed reasoning chain behind its result. This is valuable for debugging, improving prompts, or understanding model behavior.

**Please be aware:** This feature works by making an additional LLM API call for each tool invocation, which will **effectively double your token usage** for that operation.

---

## 🚀 Installation

Install the latest release via PyPI:

```bash
pip install -U hamta-texttools
```

---

## ⚡ Quick Start

```python
from openai import OpenAI

from texttools import TheTool

# Create your OpenAI client
client = OpenAI(base_url = "your_url", API_KEY = "your_api_key")

# Specify the model
model = "gpt-4o-mini"

# Create an instance of TheTool
# ⚠️ Note: Enabling `with_analysis=True` provides deeper insights but incurs additional LLM calls and token usage.
the_tool = TheTool(client = client, model = model, with_analysis = True)

# Example: Question Detection
print(the_tool.detect_question("Is this project open source?")["result"])
# Output: True

# Example: Translation
print(the_tool.translate("سلام، حالت چطوره؟")["result"])
# Output: "Hi! How are you?"
```

---

## 📚 Use Cases

Use **TextTools** when you need to:

- 🔍 **Classify** large datasets quickly without model training  
- 🌍 **Translate** and process multilingual corpora with ease  
- 🧩 **Integrate** LLMs into production pipelines (structured outputs)  
- 📊 **Analyze** large text collections using embeddings and categorization  
- ⚙️ **Automate** common text-processing tasks without reinventing the wheel  

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to **open issues, suggest new features, or submit pull requests**.  

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
