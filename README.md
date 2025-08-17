# Text Tools

<p align="center">
  <img src="https://img.shields.io/badge/TextTools-Python%20Text%20Processing-black?style=for-the-badge&logo=python&logoColor=white">
</p>


<p align="center">
  <img src="docs/logo.png" alt="Preview" width="300" height="300">
</p>


### How to Install

Install the package using:

```bash
pip install -U hamta-texttools
```


---

## What This Library Is *Not*

This is **not** a collection of low-level utilities.

To clarify: this library **does not** include things like:
- An standard `regex`
- Word normalization utilities

---

## What This Library *Provides*

This is a set of **high-level natural language processing (NLP)** tools.

Some of the features include:
- `question_detector`: Detecting if an incoming text is a question or not
- `categorizer`: No finetuning need, categorizer
- ... (Tell me what you want!)

---

## When to Use This Library

Use `texttools` when:
- You need to **process large volumes of data using OpenAI’s GPT models** via the BATCH API.
- You want to treat an **LLM as a function** in Python that outputs structured JSON or Pydantic models.
- You need to **categorize large datasets** using vector embeddings, efficiently and at scale.