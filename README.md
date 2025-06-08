## Text tools

how to use it?

install it like this

```bash
pip install -U  git+https://git.t.etratnet.ir/AI/texttools
```

---

# what this library is NOT?

this is not a set of low level tools

what do i mean when i say its not low level? this library **will not** contain:
- an standard `regex` 
- normalization words


# What it has is:

this is a set of tools for high level NLprocessing

- question_detector: detecting if an incoming text is a question or not
- categorizer: no finetuning need, categorizer
- ... (you tell me what you want)

---

## when to use it?

when you want to:
- process a lot of data using GPT from openAI, using BATCH API
- when you want to use an LLM, as a function in python, outputting structured Json or pydantic models
- when you want to categorize a lot of data, using vector embeddings
