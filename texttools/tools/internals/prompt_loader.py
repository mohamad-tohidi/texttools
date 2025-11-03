from functools import lru_cache
from pathlib import Path
import yaml


class PromptLoader:
    """
    Utility for loading and formatting YAML prompt templates.

    Responsibilities:
    - Load and parse YAML prompt definitions.
    - Select the right template (by mode, if applicable).
    - Inject variables (`{input}`, plus any extra kwargs) into the templates.
    - Return a dict with:
        {
            "main_template": "...",
            "analyze_template": "..." | None
        }
    """

    MAIN_TEMPLATE: str = "main_template"
    ANALYZE_TEMPLATE: str = "analyze_template"

    # Use lru_cache to load each file once
    @lru_cache(maxsize=32)
    def _load_templates(self, prompt_file: str, mode: str | None) -> dict[str, str]:
        """
        Loads prompt templates from YAML file with optional mode selection.
        """
        base_dir = Path(__file__).parent.parent.parent / Path("prompts")
        prompt_path = base_dir / prompt_file
        data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))

        return {
            self.MAIN_TEMPLATE: data[self.MAIN_TEMPLATE][mode]
            if mode
            else data[self.MAIN_TEMPLATE],
            self.ANALYZE_TEMPLATE: data.get(self.ANALYZE_TEMPLATE)[mode]
            if mode
            else data.get(self.ANALYZE_TEMPLATE),
        }

    def _build_format_args(self, text: str, **extra_kwargs) -> dict[str, str]:
        # Base formatting args
        format_args = {"input": text}
        # Merge extras
        format_args.update(extra_kwargs)
        return format_args

    def load(
        self, prompt_file: str, text: str, mode: str, **extra_kwargs
    ) -> dict[str, str]:
        template_configs = self._load_templates(prompt_file, mode)
        format_args = self._build_format_args(text, **extra_kwargs)

        # Inject variables inside each template
        for key in template_configs.keys():
            template_configs[key] = template_configs[key].format(**format_args)

        return template_configs
