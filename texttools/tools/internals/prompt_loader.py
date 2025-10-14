from typing import Optional
from pathlib import Path
import yaml


class PromptLoader:
    """
    Utility for loading and formatting YAML prompt templates.

    Each YAML file under `prompts/` must define at least a `main_template`,
    and optionally an `analyze_template`. These can either be a single string
    or a dictionary keyed by mode names (if `use_modes=True`).

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

    def _load_templates(
        self,
        prompts_dir: str,
        prompt_file: str,
        mode: Optional[str],
    ) -> dict[str, str]:
        prompt_path = Path(__file__).parent.parent.parent / prompts_dir / prompt_file

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        try:
            # Load the data
            data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {prompt_path}: {e}")

        return {
            "main_template": data[self.MAIN_TEMPLATE][mode]
            if mode
            else data[self.MAIN_TEMPLATE],
            "analyze_template": data.get(self.ANALYZE_TEMPLATE)[mode]
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
        self,
        prompt_file: str,
        text: str,
        mode: str,
        prompts_dir: str = "prompts",
        **extra_kwargs,
    ) -> dict[str, str]:
        template_configs = self._load_templates(prompts_dir, prompt_file, mode)
        format_args = self._build_format_args(text, **extra_kwargs)

        # Inject variables inside each template
        for key in template_configs.keys():
            template_configs[key] = template_configs[key].format(**format_args)

        return template_configs
