# Models

## What are they?
TextTools' models can be divided into two groups:

**1. Output models**  
**2. Models used for structures output**

### Output models
These models are served as the data classes used to transfer data between parts of the project. For example, `ToolOutput` model is used to store results of each tool and is returned to the user. `OperatorOutput` is used to transfer data from **Operators** to **Tools**.

### Structured output models
These models are used to be given to the LLM so that the LLM knows how to response to the user. For example, `is_question()`'s result should be a boolean. How the LLM should know that is must only produce a boolean? Here, these models are used to serve this purpose.
