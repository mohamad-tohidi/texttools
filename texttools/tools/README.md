# Tools

## Overview
This folder contains all the **tools** provided by TextTools. Each tool is implemented in TheTool/AsyncTheTool class, designed to work with **LLMs** and structured outputs (e.g., Pydantic models, JSON).

Tools are modular, easy to extend, and ready to use for common NLP tasks.

---

## Core Components

### BaseOperator, Operator/AsyncOperator Class
The base `Operator` class provides:
- LLM client integration (OpenAI)
- Prompt loading and formatting
- Structured output parsing using Pydantic models
- Optional analysis step capability
- Mode-based prompt selection

### Output Models
Structured output models defined in `output_models.py`:
- `ToolOutput` - Output model of each tool
- `StrOutput` - Simple string output
- `ListStrOutput` - List of strings
- `ListDictStrStrOutput` - List of dictionaries
- `ReasonListStrOutput` - Output with reasoning
- `CategorizerOutput` - Specialized categorization output

### Prompt Loader
The `PromptLoader` class:
- Loads YAML prompt templates from the `prompts/` folder
- Supports mode-based template selection
- Uses lru cache for efficiency
- Handles variable injection into templates
- Manages both main and analysis templates

### Formatter
The `Fromatter` class:
- Gathers diffrent formatters to format chats
- New formatters can be added easily by defining new static methods

---

## Sync vs Async
| Tool         | Style   | Use case                                    |
|--------------|---------|---------------------------------------------|
| `TheTool`    | Sync    | Simple scripts, sequential workflows        |
| `AsyncTheTool` | Async | High-throughput apps, APIs, concurrent tasks |
