# Batch

## Overview
`BatchRunner` is a lightweight Python utility that simplifies working with the [OpenAI Batch API](https://platform.openai.com/docs/guides/batch). It allows you to:

- Efficiently submit **large volumes of prompts** for processing.  
- Automatically manage **job state** (saving, resuming, clearing).  
- Enforce **structured output** with [Pydantic](https://docs.pydantic.dev/).  
- Optionally define **custom JSON schemas** for advanced validation.  
- Fetch results and handle errors gracefully.  

This tool is especially useful if you need to process thousands of inputs (NER, classification, summarization, etc.) without manually managing batch job lifecycles.