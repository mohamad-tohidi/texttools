from dataclasses import dataclass
from collections.abc import Callable

from texttools.batch.internals.utils import import_data, export_data


@dataclass
class BatchConfig:
    """
    Configuration for batch job runner.
    """

    system_prompt: str = ""
    job_name: str = ""
    input_data_path: str = ""
    output_data_filename: str = ""
    model: str = "gpt-4.1-mini"
    MAX_BATCH_SIZE: int = 100
    MAX_TOTAL_TOKENS: int = 2_000_000
    CHARS_PER_TOKEN: float = 2.7
    PROMPT_TOKEN_MULTIPLIER: int = 1_000
    BASE_OUTPUT_DIR: str = "Data/batch_entity_result"
    import_function: Callable = import_data
    export_function: Callable = export_data
    poll_interval_seconds: int = 30
    max_retries: int = 3
