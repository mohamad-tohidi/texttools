from typing import TypeVar, Any, Union
import re
import math
import logging
import random

from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI

# Base Model type for output models
T = TypeVar("T", bound=BaseModel)

ClientType = Union[OpenAI, AsyncOpenAI]

logger = logging.getLogger("texttools.base_operator")


class BaseOperator:
    # Max retry in case of failed output validation
    MAX_RETRIES = 3

    def __init__(self, client: ClientType, model: str):
        self._client = client
        self._model = model

    def _build_user_message(self, prompt: str) -> dict[str, str]:
        return {"role": "user", "content": prompt}

    def _extract_logprobs(self, completion: dict) -> list[dict[str, Any]]:
        """
        Extracts and filters token probabilities from completion logprobs.
        Skips punctuation and structural tokens, returns cleaned probability data.
        """
        logprobs_data = []

        ignore_pattern = re.compile(r'^(result|[\s\[\]\{\}",:]+)$')

        for choice in completion.choices:
            if not getattr(choice, "logprobs", None):
                logger.error("logprobs is not available for the chosen model.")
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

    def _get_retry_temp(self, base_temp: float) -> float:
        """
        Calculate temperature for retry attempts.
        """
        delta_temp = random.choice([-1, 1]) * random.uniform(0.1, 0.9)
        new_temp = base_temp + delta_temp

        return max(0.0, min(new_temp, 1.5))
