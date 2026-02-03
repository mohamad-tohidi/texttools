# Tools

## How they work?
Each tool will pass its parameters to the **Operator** class. The operator will do chat completions and will return `OperatorOutput` model which contains the data from those completions. Then, the output is processed and returned as the final results in a `ToolOutput` model.

## BatchTheTool - How it works?
The `BatchTheTool` class is a wrapper around `AsyncTheTool` that provides parallel processing capabilities. It takes a list of texts (or other inputs) and processes them concurrently using an internal semaphore for controlled concurrency. Each method creates individual tasks for every input text, then uses `asyncio.gather` to execute them in parallel while respecting the configured `max_concurrency` limit. Results are collected and returned as a list of `ToolOutput` objects.