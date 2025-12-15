from functools import lru_cache
from pathlib import Path
import yaml

from texttools.tools.internals.exceptions import PromptError


class PromptLoader:
    """
    Utility for loading and formatting YAML prompt templates.

    Responsibilities:
    - Load and parse YAML prompt definitions.
    - Select the right template (by mode, if applicable).
    - Inject variables (`{input}`, plus any extra kwargs) into the templates.
    """

    MAIN_TEMPLATE = "main_template"
    ANALYZE_TEMPLATE = "analyze_template"

    @staticmethod
    def _build_format_args(text: str, **extra_kwargs) -> dict[str, str]:
        # Base formatting args
        format_args = {"input": text}
        # Merge extras
        format_args.update(extra_kwargs)
        return format_args

    # Use lru_cache to load each file once
    @lru_cache(maxsize=32)
    def _load_templates(self, prompt_file: str, mode: str | None) -> dict[str, str]:
        """
        Loads prompt templates from YAML file with optional mode selection.
        """
        try:
            base_dir = Path(__file__).parent.parent.parent / Path("prompts")
            prompt_path = base_dir / prompt_file

            if not prompt_path.exists():
                raise PromptError(f"Prompt file not found: {prompt_file}")

            data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))

            if self.MAIN_TEMPLATE not in data:
                raise PromptError(f"Missing 'main_template' in {prompt_file}")

            if mode and mode not in data.get(self.MAIN_TEMPLATE, {}):
                raise PromptError(f"Mode '{mode}' not found in {prompt_file}")

            return {
                self.MAIN_TEMPLATE: data[self.MAIN_TEMPLATE][mode]
                if mode and isinstance(data[self.MAIN_TEMPLATE], dict)
                else data[self.MAIN_TEMPLATE],
                self.ANALYZE_TEMPLATE: data.get(self.ANALYZE_TEMPLATE, {}).get(mode)
                if mode and isinstance(data.get(self.ANALYZE_TEMPLATE), dict)
                else data.get(self.ANALYZE_TEMPLATE, ""),
            }

        except yaml.YAMLError as e:
            raise PromptError(f"Invalid YAML in {prompt_file}: {e}")
        except Exception as e:
            raise PromptError(f"Failed to load prompt {prompt_file}: {e}")

    def load(
        self, prompt_file: str, text: str, mode: str, **extra_kwargs
    ) -> dict[str, str]:
        try:
            template_configs = self._load_templates(prompt_file, mode)
            format_args = self._build_format_args(text, **extra_kwargs)

            # Inject variables inside each template
            for key in template_configs.keys():
                template_configs[key] = template_configs[key].format(**format_args)

            return template_configs

        except KeyError as e:
            raise PromptError(f"Missing template variable: {e}")
        except Exception as e:
            raise PromptError(f"Failed to format prompt: {e}")
