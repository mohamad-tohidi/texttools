import json
import os
import time
from pathlib import Path
from typing import Type, TypeVar
import logging

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from texttools.batch.batch_manager import BatchManager
from texttools.batch.batch_config import BatchConfig
from texttools.internals.models import Str
from texttools.internals.exceptions import TextToolsError, ConfigurationError

# Base Model type for output models
T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("texttools.batch_runner")


class BatchJobRunner:
    """
    Handles running batch jobs using a batch manager and configuration.
    """

    def __init__(
        self, config: BatchConfig = BatchConfig(), output_model: Type[T] = Str
    ):
        try:
            self._config = config
            self._system_prompt = config.system_prompt
            self._job_name = config.job_name
            self._input_data_path = config.input_data_path
            self._output_data_filename = config.output_data_filename
            self._model = config.model
            self._output_model = output_model
            self._manager = self._init_manager()
            self._data = self._load_data()
            self._parts: list[list[dict[str, object]]] = []
            # Map part index to job name
            self._part_idx_to_job_name: dict[int, str] = {}
            # Track retry attempts per part
            self._part_attempts: dict[int, int] = {}
            self._partition_data()
            Path(self._config.BASE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        except Exception as e:
            raise ConfigurationError(f"Batch runner initialization failed: {e}")

    def _init_manager(self) -> BatchManager:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        return BatchManager(
            client=client,
            model=self._model,
            prompt_template=self._system_prompt,
            output_model=self._output_model,
        )

    def _load_data(self):
        with open(self._input_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = self._config.export_function(data)

        # Ensure data is a list of dicts with 'id' and 'content' as strings
        if not isinstance(data, list):
            raise ValueError(
                "Exported data must be a list of dicts with 'id' and 'content' keys"
            )
        for item in data:
            if not (isinstance(item, dict) and "id" in item and "content" in item):
                raise ValueError(
                    f"Item must be a dict with 'id' and 'content' keys. Got: {type(item)}"
                )
            if not (isinstance(item["id"], str) and isinstance(item["content"], str)):
                raise ValueError("'id' and 'content' must be strings.")
        return data

    def _partition_data(self):
        total_length = sum(len(item["content"]) for item in self._data)
        prompt_length = len(self._system_prompt)
        total = total_length + (prompt_length * len(self._data))
        calculation = total / self._config.CHARS_PER_TOKEN
        logger.info(
            f"Total chars: {total_length}, Prompt chars: {prompt_length}, Total: {total}, Tokens: {calculation}"
        )
        if calculation < self._config.MAX_TOTAL_TOKENS:
            self._parts = [self._data]
        else:
            # Partition into chunks of MAX_BATCH_SIZE
            self._parts = [
                self._data[i : i + self._config.MAX_BATCH_SIZE]
                for i in range(0, len(self._data), self._config.MAX_BATCH_SIZE)
            ]
        logger.info(f"Data split into {len(self._parts)} part(s)")

    def _submit_all_jobs(self) -> None:
        for idx, part in enumerate(self._parts):
            if self._result_exists(idx):
                logger.info(f"Skipping part {idx + 1}: result already exists.")
                continue
            part_job_name = (
                f"{self._job_name}_part_{idx + 1}"
                if len(self._parts) > 1
                else self._job_name
            )
            # If a job with this name already exists, register and skip submitting
            existing_job = self._manager._load_state(part_job_name)
            if existing_job:
                logger.info(
                    f"Skipping part {idx + 1}: job already exists ({part_job_name})."
                )
                self._part_idx_to_job_name[idx] = part_job_name
                self._part_attempts.setdefault(idx, 0)
                continue

            payload = part
            logger.info(
                f"Submitting job for part {idx + 1}/{len(self._parts)}: {part_job_name}"
            )
            self._manager.start(payload, job_name=part_job_name)
            self._part_idx_to_job_name[idx] = part_job_name
            self._part_attempts.setdefault(idx, 0)
            # This is added for letting file get uploaded, before starting the next part.
            logger.info("Uploading...")
            time.sleep(30)

    def _save_results(
        self,
        output_data: list[dict[str, object]] | dict[str, object],
        log: list[object],
        part_idx: int,
    ):
        part_suffix = f"_part_{part_idx + 1}" if len(self._parts) > 1 else ""
        result_path = (
            Path(self._config.BASE_OUTPUT_DIR)
            / f"{Path(self._output_data_filename).stem}{part_suffix}.json"
        )
        if not output_data:
            logger.info("No output data to save. Skipping this part.")
            return
        else:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
        if log:
            log_path = (
                Path(self._config.BASE_OUTPUT_DIR)
                / f"{Path(self._output_data_filename).stem}{part_suffix}_log.json"
            )
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log, f, ensure_ascii=False, indent=4)

    def _result_exists(self, part_idx: int) -> bool:
        part_suffix = f"_part_{part_idx + 1}" if len(self._parts) > 1 else ""
        result_path = (
            Path(self._config.BASE_OUTPUT_DIR)
            / f"{Path(self._output_data_filename).stem}{part_suffix}.json"
        )
        return result_path.exists()

    def run(self):
        """
        Execute the batch job processing pipeline.

        Submits jobs, monitors progress, handles retries, and saves results.
        """
        try:
            # Submit all jobs up-front for concurrent execution
            self._submit_all_jobs()
            pending_parts: set[int] = set(self._part_idx_to_job_name.keys())
            logger.info(f"Pending parts: {sorted(pending_parts)}")
            # Polling loop
            while pending_parts:
                finished_this_round: list[int] = []
                for part_idx in list(pending_parts):
                    job_name = self._part_idx_to_job_name[part_idx]
                    status = self._manager.check_status(job_name=job_name)
                    logger.info(f"Status for {job_name}: {status}")
                    if status == "completed":
                        logger.info(
                            f"Job completed. Fetching results for part {part_idx + 1}..."
                        )
                        output_data, log = self._manager.fetch_results(
                            job_name=job_name, remove_cache=False
                        )
                        output_data = self._config.import_function(output_data)
                        self._save_results(output_data, log, part_idx)
                        logger.info(
                            f"Fetched and saved results for part {part_idx + 1}."
                        )
                        finished_this_round.append(part_idx)
                    elif status == "failed":
                        attempt = self._part_attempts.get(part_idx, 0) + 1
                        self._part_attempts[part_idx] = attempt
                        if attempt <= self._config.max_retries:
                            logger.info(
                                f"Job {job_name} failed (attempt {attempt}). Retrying after short backoff..."
                            )
                            self._manager._clear_state(job_name)
                            time.sleep(10)
                            payload = self._to_manager_payload(self._parts[part_idx])
                            new_job_name = (
                                f"{self._job_name}_part_{part_idx + 1}_retry_{attempt}"
                            )
                            self._manager.start(payload, job_name=new_job_name)
                            self._part_idx_to_job_name[part_idx] = new_job_name
                        else:
                            logger.info(
                                f"Job {job_name} failed after {attempt - 1} retries. Marking as failed."
                            )
                            finished_this_round.append(part_idx)
                    else:
                        # Still running or queued
                        continue
                # Remove finished parts
                for part_idx in finished_this_round:
                    pending_parts.discard(part_idx)
                if pending_parts:
                    logger.info(
                        f"Waiting {self._config.poll_interval_seconds}s before next status check for parts: {sorted(pending_parts)}"
                    )
                    time.sleep(self._config.poll_interval_seconds)

        except Exception as e:
            raise TextToolsError(f"Batch job execution failed: {e}")
