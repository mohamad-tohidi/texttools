# Tools Folder

## Overview
This folder contains all the **tools** provided by TextTools. Each tool is implemented as a Python class, designed to work with **LLMs** and structured outputs (e.g., Pydantic models, JSON).

Tools are modular, easy to extend, and ready to use for common NLP tasks.

## Structure
- **tool_name/**: Each subfolder contains the implementation of a specific tool.
- **tool_name.py**: Main Python class implementing the tool.

### Example Folder Layout
```
tools/
├─ categorizer/
│  └─ categorizer.py
├─ question_merger/
│  └─ question_merger.py
├─ translator/
│  └─ translator.py
└─ ...
```

## Guidelines
1. **Naming**: Each tool folder should be named after the tool, e.g., `question_merger`, `translator`.
2. **Consistency**: Use a consistent interface with `__init__`, `run()` or similar methods.
3. **Output Models**: Use Pydantic models for structured outputs.
4. **Prompts**: Tools that rely on prompts should load them dynamically from the `prompts/` folder.
