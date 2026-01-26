# Tools

## How they work?
Each tool will pass its parametrs to the **Operator** class. The operator will do chat completions and will return `OperatorOutput` model which contains the data from those completions. Then, the output is processed and returned as the final results in a `ToolOutput` model.