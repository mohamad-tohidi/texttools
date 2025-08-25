from typing import Optional, Type, TypeVar
import random
import asyncio

from openai import OpenAI, RateLimitError, APIError
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


async def flex_processing(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    output_model: Type[T] = None,
    prompt_cache_key: Optional[str] = None,
    max_retries: int = 10,
    base_delay: float = 2.0,
    model: Optional[str] = "gpt-5-mini",
    **client_kwargs,
):
    """
    Wrapper for flex processing with retry and exponential backoff.
    Handles 429 'Resource Unavailable' errors gracefully.
    """
    for attempt in range(max_retries):
        try:
            request_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "service_tier": "flex",
                "timeout": 900.0,
                **client_kwargs,
            }

            if output_model:
                request_kwargs["response_format"] = output_model
            if prompt_cache_key:
                request_kwargs["prompt_cache_key"] = prompt_cache_key

            response = client.chat.completions.parse(**request_kwargs)
            content = response.choices[0].message.content

            # Validate structured output if a model is provided
            if output_model is not None:
                try:
                    output_model.model_validate_json(content)
                    base_content = output_model(**content)
                    return base_content
                except ValidationError as ve:
                    # Treat invalid output as retryable
                    wait_time = base_delay * (2**attempt) + random.uniform(0, 1)
                    print(
                        f"[Flex Retry] Attempt {attempt + 1}/{max_retries} produced invalid structured output."
                        f"Retrying in {wait_time:.2f}s... (ValidationError: {ve})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

                # Valid response
                return content

        except (RateLimitError, APIError) as e:
            wait_time = base_delay * (2**attempt) + random.uniform(0, 1)
            print(
                f"[Flex Retry] Attempt {attempt + 1}/{max_retries} failed "
                f"with error: {type(e).__name__} - {e}. "
                f"Retrying in {wait_time:.2f}s..."
            )
            await asyncio.sleep(wait_time)

        except Exception as e:
            # Non-recoverable error: break out immediately
            raise RuntimeError(
                f"[Flex Processing] Unrecoverable error for prompt_key={prompt_cache_key}: {e}"
            )

    raise RuntimeError(
        f"[Flex Processing] Exhausted {max_retries} retries for prompt_key={prompt_cache_key}"
    )
