# Formatters

## Overview
This folder contains formatter classes used to structure inputs before sending them to different AI model providers.  
Each formatter is responsible for transforming raw messages or text into a standardized payload format.

---

## Structure
- **user_merge_formatter.py**: Implements the `UserMergeFormatter`, which merges consecutive user messages and replaces system roles with user roles.

### UserMergeFormatter
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
