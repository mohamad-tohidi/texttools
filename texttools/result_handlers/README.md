# Result Handlers

## Overview
This folder contains result handler classes used to process the outputs of tools or AI models.  
Each handler defines a custom strategy for dealing with results (e.g., saving to file, printing, or ignoring).

## Structure
- **base_result_handler.py**: Defines the abstract `BaseResultHandler` class. All custom handlers must inherit from this.
- **save_to_file_result_handler.py**: Implements `SaveToFileResultHandler`, which writes results to a file.
- **print_result_handler.py**: Implements `PrintResultHandler`, which prints results to the console.
- **no_op_result_handler.py**: Implements `NoOpResultHandler`, which ignores results (useful as a default).

### BaseResultHandler
- Provides a common interface for all result handlers.
- Abstract method:
  ```python
  def handle(self, results: dict[str, Any]) -> None
  ```
- Subclasses implement how results are processed (stored, printed, ignored, etc.).

### SaveToFileResultHandler
- Saves results to a file in CSV-like format.
- Serializes objects into JSON (via Pydantic if available, otherwise `json.dumps`, otherwise `str`).
- Example output format:
  ```csv
  input_text,result_value
  ```

### PrintResultHandler
- Prints the result value to the console.
- Useful for debugging or quick experiments.

### NoOpResultHandler
- Does nothing with results.
- Useful as a fallback or when no processing is required.

### Example Usage
```python
results = {"input_text": "Hello", "result": {"answer": "Hi there!"}}

# Save to file
file_handler = SaveToFileResultHandler("results.csv")
file_handler.handle(results)

# Print to console
print_handler = PrintResultHandler()
print_handler.handle(results)

# Do nothing
no_op_handler = NoOpResultHandler()
no_op_handler.handle(results)
```

## Guidelines
1. **Inheritance**: All new handlers must inherit from `BaseResultHandler`.
2. **Serialization**: When saving results, ensure that all values are converted to strings or JSON-serializable objects.
3. **Flexibility**: Handlers should be minimal and focused on a single type of processing.
4. **Extensibility**: New handlers can be added to support additional result-processing needs (e.g., databases, APIs, logging systems).
