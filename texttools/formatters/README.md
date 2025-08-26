# Formatters

## Overview
This folder contains formatter classes used to structure inputs before sending them to different AI model providers.  
Each formatter is responsible for transforming raw messages or text into a standardized payload format.

## Structure
- **base_formatter.py**: Defines the abstract `BaseFormatter` class. All custom formatters should inherit from this.
- **user_merge_formatter.py**: Implements the `UserMergeFormatter`, which merges consecutive user messages and replaces system roles with user roles.

### BaseFormatter
- Defines a common interface for all formatters.
- Abstract method:
  ```python
  def format(self, text: str, reason: Optional[str], schema_instr: str, prompt_template: Optional[str]) -> Any
  ```
- Implementations decide the final shape (string, list of role/content dicts, or provider-specific JSON).

### UserMergeFormatter
- Inherits from `BaseFormatter`.
- Features:
  - Merges consecutive `user` messages into a single block separated by newlines.
  - Converts `system` role to `user` role.
  - Keeps `assistant` messages unchanged.
  - Validates that each message has the correct keys (`role`, `content`) and valid roles.

### Example Usage
```python
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm fine, thanks."},
]

formatter = UserMergeFormatter()
formatted = formatter.format(messages)

# Result:
# [
#   {"role": "user", "content": "Hello\nHow are you?"},
#   {"role": "assistant", "content": "I'm fine, thanks."}
# ]
```

## Guidelines
1. **Inheritance**: New formatters must inherit from `BaseFormatter`.
2. **Validation**: Each formatter should validate inputs before transforming them.
3. **Consistency**: Ensure that role and content fields remain consistent across all implementations.
4. **Extensibility**: New formatters can be added to handle other providers or formatting needs.
