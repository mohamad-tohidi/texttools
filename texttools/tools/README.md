# Tools

## Overview
This folder contains all the **tools** provided by TextTools. Each tool is implemented in TheTool class, designed to work with **LLMs** and structured outputs (e.g., Pydantic models, JSON).

Tools are modular, easy to extend, and ready to use for common NLP tasks.

## Available Tools

The `TheTool` class provides the following NLP operations:

- **`categorize()`** - Classifies text into Islamic studies categories (باورهای دینی, اخلاق اسلامی, etc.)
- **`detect_question()`** - Binary detection of whether input is a question
- **`extract_keywords()`** - Extracts keywords from text
- **`extract_entities()`** - Named Entity Recognition (NER) system
- **`summarize()`** - Text summarization
- **`generate_question_from_text()`** - Generates questions from text
- **`merge_questions()`** - Merges multiple questions with different modes
- **`rewrite_question()`** - Rewrites questions with different wording/meaning
- **`generate_questions_from_subject()`** - Generates questions about a specific subject
- **`translate()`** - Text translation between languages
- **`custom_tool()`** - Allows users to define a custom tool with arbitrary BaseModel

## Architecture

### Core Components

#### Operator Class
The base `Operator` class provides:
- LLM client integration (OpenAI)
- Prompt loading and formatting
- Structured output parsing using Pydantic models
- Optional analysis step capability
- Mode-based prompt selection

#### Output Models
Structured output models defined in `output_models.py`:
- `StrOutput` - Simple string output
- `ListStrOutput` - List of strings
- `ListDictStrStrOutput` - List of dictionaries
- `ReasonListStrOutput` - Output with reasoning
- `CategorizerOutput` - Specialized categorization output

#### Prompt Loader
The `PromptLoader` class:
- Loads YAML prompt templates from the `prompts/` folder
- Supports mode-based template selection
- Handles variable injection into templates
- Manages both main and analysis templates

## Sync vs Async
| Tool         | Style   | Use case                                    |
|--------------|---------|---------------------------------------------|
| `TheTool`    | Sync    | Simple scripts, sequential workflows        |
| `AsyncTheTool` | Async | High-throughput apps, APIs, concurrent tasks |

## Usage Example

```python
from openai import OpenAI
from texttools import TheTool

# Initialize client and tool
client = OpenAI(base_url="your-base-url", api_key="your-api-key")
tool = TheTool(client=client, model="gpt-4")

# Use any tool
result = tool.categorize("نمازهای یومیه چگونه خوانده می‌شوند؟")
print(result["result"])

# With analysis
result = tool.extract_keywords("متن نمونه برای استخراج کلمات کلیدی", with_analysis=True)
print(result["result"])
print(result["analysis"])  # Available when with_analysis=True
```
