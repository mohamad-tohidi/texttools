# Tools

## Overview
This folder contains all the **tools** provided by TextTools. Each tool is implemented in TheTool/AsyncTheTool class, designed to work with **LLMs** and structured outputs (e.g., Pydantic models, JSON).

Tools are modular, easy to extend, and ready to use for common NLP tasks.

---

## Core Components

### OperatorUtils, Operator/AsyncOperator Class
The `Operator` class provides:
- LLM client integration (OpenAI)
- Prompt loading and formatting
- Structured output parsing using Pydantic models
- Optional analysis step capability
- Mode-based prompt selection
- Inject user prompts
- Extract logprobs

### Models
Models defined in `models.py`:
- `ToolOutput` - Output model of each tool
- `StrOutput` - Simple string output
- `ListStrOutput` - List of strings
- `ListDictStrStrOutput` - List of dictionaries
- `ReasonListStrOutput` - Output with reasoning
- `CategoryTree` - Category tree for categorizer
- `create_dynamic_model() - Creates dynamic BaseModel for categorizer


### Formatter
The `Fromatter` class:
- Gathers diffrent formatters to format chats
- New formatters can be added easily by defining new static methods

### Prompt Loader
The `PromptLoader` class:
- Loads YAML prompt templates from the `prompts/` folder
- Supports mode-based template selection
- Uses lru cache for efficiency
- Handles variable injection into templates
- Manages both main and analysis templates
