# Operators

## What are they?
**Operators** are like the engine of TextTools. They run openai chat completions, create the prompts, etc. Their input is the data given by `TheTool`/`AsyncTheTool` which calls them. The operators will do the chat completions and return a `OperatorOutput` as the result. This output will be processed and will be returned to the user as the final output of each tool.