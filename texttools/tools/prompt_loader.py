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

    def __init__(self, prompts_dir: Optional[str] = None):
        self.PROMPTS_DIR = prompts_dir or "prompts"

    def _get_prompt_path(self, prompt_file: str) -> Path:
        return Path(__file__).parent.parent / self.PROMPTS_DIR / prompt_file

    def _load_templates(
        self, prompt_file: str, use_modes: bool, mode: str
    ) -> dict[str, str]:
        prompt_path = self._get_prompt_path(prompt_file)

        # Error handlings
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        try:
            data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {prompt_path}: {e}")

        if self.MAIN_TEMPLATE not in data:
            raise ValueError(
                f"Missing required '{self.MAIN_TEMPLATE}' in {prompt_file}"
            )

        # Load the data
        data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))

        return {
            self.MAIN_TEMPLATE: data[self.MAIN_TEMPLATE][mode]
            if use_modes
            else data[self.MAIN_TEMPLATE],
            self.ANALYZE_TEMPLATE: data.get(self.ANALYZE_TEMPLATE)[mode]
            if use_modes
            else data.get(self.ANALYZE_TEMPLATE),
        }

    def _build_format_args(self, input_text: str, **extra_kwargs) -> dict[str, str]:
        # Base formatting args
        format_args = {"input": input_text}
        # Merge extras
        format_args.update(extra_kwargs)
        return format_args

    def load_prompts(
        self,
        prompt_file: str,
        use_modes: bool,
        mode: str,
        input_text: str,
        **extra_kwargs,
    ) -> dict[str, str]:
        template_configs = self._load_templates(prompt_file, use_modes, mode)
        format_args = self._build_format_args(input_text, **extra_kwargs)

        # Inject variables inside each template
        for key in template_configs.keys():
            template_configs[key] = template_configs[key].format(**format_args)

        return template_configs
