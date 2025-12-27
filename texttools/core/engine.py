import asyncio
import math
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .exceptions import PromptError


class PromptLoader:
    """
    Utility for loading and formatting YAML prompt templates.
    """

    MAIN_TEMPLATE = "main_template"
    ANALYZE_TEMPLATE = "analyze_template"

    @lru_cache(maxsize=32)
    def _load_templates(self, prompt_file: str, mode: str | None) -> dict[str, str]:
        try:
            base_dir = Path(__file__).parent.parent / Path("prompts")
            prompt_path = base_dir / prompt_file

            if not prompt_path.exists():
                raise PromptError(f"Prompt file not found: {prompt_file}")

            data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))

            if self.MAIN_TEMPLATE not in data:
                raise PromptError(f"Missing 'main_template' in {prompt_file}")

            if self.ANALYZE_TEMPLATE not in data:
                raise PromptError(f"Missing 'analyze_template' in {prompt_file}")

            if mode and mode not in data.get(self.MAIN_TEMPLATE, {}):
                raise PromptError(f"Mode '{mode}' not found in {prompt_file}")

            main_template = (
                data[self.MAIN_TEMPLATE][mode]
                if mode and isinstance(data[self.MAIN_TEMPLATE], dict)
                else data[self.MAIN_TEMPLATE]
            )

            analyze_template = (
                data[self.ANALYZE_TEMPLATE][mode]
                if mode and isinstance(data[self.ANALYZE_TEMPLATE], dict)
                else data[self.ANALYZE_TEMPLATE]
            )

            if not main_template or not main_template.strip():
                raise PromptError(
                    f"Empty main_template in {prompt_file}"
                    + (f" for mode '{mode}'" if mode else "")
                )

            return {
                self.MAIN_TEMPLATE: main_template,
                self.ANALYZE_TEMPLATE: analyze_template,
            }

        except yaml.YAMLError as e:
            raise PromptError(f"Invalid YAML in {prompt_file}: {e}")
        except Exception as e:
            raise PromptError(f"Failed to load prompt {prompt_file}: {e}")

    def load(
        self, prompt_file: str, text: str, mode: str, **extra_kwargs
    ) -> dict[str, str]:
        try:
            format_args = {"text": text}
            format_args.update(extra_kwargs)

            template_configs = self._load_templates(prompt_file, mode)
            for key, value in template_configs.items():
                template_configs[key] = value.format(**format_args)

            return template_configs

        except KeyError as e:
            raise PromptError(f"Missing template variable: {e}")
        except Exception as e:
            raise PromptError(f"Failed to format prompt: {e}")


class OperatorUtils:
    @staticmethod
    def build_main_prompt(
        main_template: str,
        analysis: str | None,
        output_lang: str | None,
        user_prompt: str | None,
    ) -> str:
        parts = []

        if analysis:
            parts.append(f"Based on this analysis: {analysis}")
        if output_lang:
            parts.append(f"Respond only in the {output_lang} language.")
        if user_prompt:
            parts.append(f"Consider this instruction: {user_prompt}")

        parts.append(main_template)
        return "\n".join(parts)

    @staticmethod
    def build_message(prompt: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def extract_logprobs(completion: Any) -> list[dict]:
        """
        Extracts and filters logprobs from completion.
        Skips punctuation and structural tokens.
        """
        logprobs_data = []

        ignore_pattern = re.compile(r'^(result|[\s\[\]\{\}",:]+)$')

        for choice in completion.choices:
            if not getattr(choice, "logprobs", None):
                raise ValueError("Your model does not support logprobs")

            for logprob_item in choice.logprobs.content:
                if ignore_pattern.match(logprob_item.token):
                    continue
                token_entry = {
                    "token": logprob_item.token,
                    "prob": round(math.exp(logprob_item.logprob), 8),
                    "top_alternatives": [],
                }
                for alt in logprob_item.top_logprobs:
                    if ignore_pattern.match(alt.token):
                        continue
                    token_entry["top_alternatives"].append(
                        {
                            "token": alt.token,
                            "prob": round(math.exp(alt.logprob), 8),
                        }
                    )
                logprobs_data.append(token_entry)

        return logprobs_data

    @staticmethod
    def get_retry_temp(base_temp: float) -> float:
        new_temp = base_temp + random.choice([-1, 1]) * random.uniform(0.1, 0.9)
        return max(0.0, min(new_temp, 1.5))


def text_to_chunks(text: str, size: int, overlap: int) -> list[str]:
    """
    Utility for chunking large texts. Used for translation tool
    """
    separators = ["\n\n", "\n", " ", ""]
    is_separator_regex = False
    keep_separator = True
    length_function = len
    strip_whitespace = True
    chunk_size = size
    chunk_overlap = overlap

    def _split_text_with_regex(
        text: str, separator: str, keep_separator: bool
    ) -> list[str]:
        if not separator:
            return [text]
        if not keep_separator:
            return re.split(separator, text)
        _splits = re.split(f"({separator})", text)
        splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
        if len(_splits) % 2 == 0:
            splits += [_splits[-1]]
        return [_splits[0]] + splits if _splits[0] else splits

    def _join_docs(docs: list[str], separator: str) -> str | None:
        text = separator.join(docs)
        if strip_whitespace:
            text = text.strip()
        return text if text else None

    def _merge_splits(splits: list[str], separator: str) -> list[str]:
        separator_len = length_function(separator)
        docs = []
        current_doc = []
        total = 0
        for d in splits:
            len_ = length_function(d)
            if total + len_ + (separator_len if current_doc else 0) > chunk_size:
                if total > chunk_size:
                    pass
                if current_doc:
                    doc = _join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    while total > chunk_overlap or (
                        total + len_ + (separator_len if current_doc else 0)
                        > chunk_size
                        and total > 0
                    ):
                        total -= length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += len_ + (separator_len if len(current_doc) > 1 else 0)
        doc = _join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def _split_text(text: str, separators: list[str]) -> list[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            separator_ = _s if is_separator_regex else re.escape(_s)
            if not _s:
                separator = _s
                break
            if re.search(separator_, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break
        separator_ = separator if is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, separator_, keep_separator)
        _separator = "" if keep_separator else separator
        good_splits = []
        for s in splits:
            if length_function(s) < chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = _merge_splits(good_splits, _separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = _split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if good_splits:
            merged_text = _merge_splits(good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    return _split_text(text, separators)


async def run_with_timeout(coro, timeout: float | None):
    """
    Utility for timeout logic defined in AsyncTheTool
    """
    if timeout is None:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation exceeded timeout of {timeout} seconds")
