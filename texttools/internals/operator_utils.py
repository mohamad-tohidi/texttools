import re
import math
import random


class OperatorUtils:
    @staticmethod
    def build_main_prompt(
        main_template: str,
        analysis: str | None,
        output_lang: str | None,
        user_prompt: str | None,
    ) -> str:
        main_prompt = ""

        if analysis:
            main_prompt += f"Based on this analysis:\n{analysis}\n"

        if output_lang:
            main_prompt += f"Respond only in the {output_lang} language.\n"

        if user_prompt:
            main_prompt += f"Consider this instruction {user_prompt}\n"

        main_prompt += main_template

        return main_prompt

    @staticmethod
    def build_message(prompt: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def extract_logprobs(completion: dict) -> list[dict]:
        """
        Extracts and filters token probabilities from completion logprobs.
        Skips punctuation and structural tokens, returns cleaned probability data.
        """
        logprobs_data = []

        ignore_pattern = re.compile(r'^(result|[\s\[\]\{\}",:]+)$')

        for choice in completion.choices:
            if not getattr(choice, "logprobs", None):
                return []

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
        """
        Calculate temperature for retry attempts.
        """
        delta_temp = random.choice([-1, 1]) * random.uniform(0.1, 0.9)
        new_temp = base_temp + delta_temp

        return max(0.0, min(new_temp, 1.5))
