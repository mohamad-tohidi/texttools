from pathlib import Path
import yaml


class PromptLoader:
    """
    Loads YAML with both `main_template` and `analyze_template`
    """

    MAIN_TEMPLATE: str = "main_template"
    ANALYZE_TEMPLATE: str = "analyze_template"
    PROMPTS_DIR: str = "prompts"

    def _load_templates(
        self, prompt_file: str, use_modes: bool, mode: str
    ) -> dict[str, str]:
        prompt_file = Path(__file__).parent.parent / self.PROMPTS_DIR / prompt_file

        data = yaml.safe_load(prompt_file.read_text(encoding="utf-8"))

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
