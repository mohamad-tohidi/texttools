# Prompts

## What are they?
Each prompt is a yaml file used to generate response from the LLM. They are opened and formatted by `load_prompt()` function which is in `OperatorUtils` class.

Some prompt files have two or more **Modes**. In this project, modes are defined to prevent adding a lot of tools. For example, in `TheTool`, there is `augment()` tool which has three modes: **positive, negative and hard_negative**. Modes are used not to create separate tools for each kind of text augmentation. Each mode has its own prompt, but not in separate tools, they are gathered in `augment.yaml` file for simplicity. 

## Prompt quality
Every prompt is well-engineered. They have a consistent format, and are carefully written to get the best output from LLMs.