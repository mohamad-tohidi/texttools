# Internals

## Core Components

### OperatorUtils, Operator, AsyncOperator
The `Operator` class provides:
- LLM client integration (OpenAI)
- Prompt loading and formatting
- Structured output parsing using Pydantic models
- Optional analysis step capability
- Mode-based prompt selection
- Inject user prompts
- Extract logprobs
- Format prompts

### Models
Models defined in `models.py`:
- `ToolOutput` - Output model of each tool
- `Str` - Simple string output
- `ListStr` - List of strings
- `ListDictStrStr` - List of dictionaries
- `ReasonListStr` - Output with reasoning
- `CategoryTree` - Category tree for categorizer
- `create_dynamic_model() - Creates dynamic BaseModel for categorizer

### Prompt Loader
The `PromptLoader` class:
- Loads YAML prompt templates from the `prompts/` folder
- Supports mode-based template selection
- Uses lru cache for efficiency
- Handles variable injection into templates
- Manages both main and analysis templates

### Exceptions
TextTools exceptions!
