# Prompts

## Overview
This folder contains YAML files for all prompts used in the project. Each file represents a separate prompt template, which can be loaded by tools or scripts that require structured prompts for AI models.

---

## Structure
- **prompt_file.yaml**: Each YAML file represents a single prompt template.
- **main_template**: The main instruction template for the model.
- **analyze_template** (optional): A secondary reasoning template used before generating the final response.
- **Modes** (optional): Some prompts may have multiple modes (e.g., `default`, `reason`) to allow different behaviors.

### Example YAML Structure
```yaml
main_template:
  default: |
    Your main instructions here with placeholders like {input}.
  reason: |
    Optional reasoning instructions here.

analyze_template:
  default: |
    Analyze and summarize the input.
  reason: |
    Optional detailed analysis template.
```

---

## Guidelines
1. **Naming**: Use descriptive names for each YAML file corresponding to the tool or task it serves.
2. **Placeholders**: Use `{input}` or other relevant placeholders to dynamically inject data.
3. **Modes**: If using modes, ensure both `main_template` and `analyze_template` contain the corresponding keys.
4. **Consistency**: Keep formatting consistent across files for easier parsing by scripts.
