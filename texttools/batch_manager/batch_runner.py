import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from texttools.batch_manager import SimpleBatchManager


class OutputModel(BaseModel):
    desired_output: str


def exporting_data(data):
    """
    Produces a structure of the following form from an initial data structure:
    [
        {"id": str, "content": str},...
    ]
    """
    return data


def importing_data(data):
    """
    Takes the output and adds and aggregates it to the original structure.
    """
    return data


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
    MAX_TOTAL_TOKENS: int = 2000000
    CHARS_PER_TOKEN: float = 2.7
    PROMPT_TOKEN_MULTIPLIER: int = 1000
    BASE_OUTPUT_DIR: str = "Data/batch_entity_result"
    import_function: Callable = importing_data
    export_function: Callable = exporting_data


class BatchJobRunner:
    """
    Handles running batch jobs using a batch manager and configuration.
    """

    def __init__(
        self, config: BatchConfig = BatchConfig(), output_model: type = OutputModel
    ):
        self.config = config
        self.system_prompt = config.system_prompt
        self.job_name = config.job_name
        self.input_data_path = config.input_data_path
        self.output_data_filename = config.output_data_filename
        self.model = config.model
        self.output_model = output_model
        self.manager = self._init_manager()
        self.data = self._load_data()
        self.parts: list[list[dict[str, Any]]] = []
        self._partition_data()
        Path(self.config.BASE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def _init_manager(self) -> SimpleBatchManager:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        return SimpleBatchManager(
            client=client,
            model=self.model,
            prompt_template=self.system_prompt,
            output_model=self.output_model,
        )

    def _load_data(self):
        with open(self.input_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = self.config.export_function(data)
        # Validation: ensure data is a list of dicts with 'id' and 'content' as strings
        if not isinstance(data, list):
            raise ValueError(
                'Exported data must be a list in this form:  [ {"id": str, "content": str},...]'
            )
        for item in data:
            if not (isinstance(item, dict) and "id" in item and "content" in item):
                raise ValueError(
                    "Each item must be a dict with 'id' and 'content' keys."
                )
            if not (isinstance(item["id"], str) and isinstance(item["content"], str)):
                raise ValueError("'id' and 'content' must be strings.")
        return data

    def _partition_data(self):
        total_length = sum(len(item["content"]) for item in self.data)
        prompt_length = len(self.system_prompt)
        total = total_length + (prompt_length * len(self.data))
        calculation = total / self.config.CHARS_PER_TOKEN
        print(
            f"Total chars: {total_length}, Prompt chars: {prompt_length}, Total: {total}, Tokens: {calculation}"
        )
        if calculation < self.config.MAX_TOTAL_TOKENS:
            self.parts = [self.data]
        else:
            # Partition into chunks of MAX_BATCH_SIZE
            self.parts = [
                self.data[i : i + self.config.MAX_BATCH_SIZE]
                for i in range(0, len(self.data), self.config.MAX_BATCH_SIZE)
            ]
        print(f"Data split into {len(self.parts)} part(s)")

    def run(self):
        for idx, part in enumerate(self.parts):
            if self._result_exists(idx):
                print(f"Skipping part {idx + 1}: result already exists.")
                continue
            part_job_name = (
                f"{self.job_name}_part_{idx + 1}"
                if len(self.parts) > 1
                else self.job_name
            )
            print(
                f"\n--- Processing part {idx + 1}/{len(self.parts)}: {part_job_name} ---"
            )
            self._process_part(part, part_job_name, idx)

    def _process_part(
        self, part: list[dict[str, Any]], part_job_name: str, part_idx: int
    ):
        while True:
            print(f"Starting job for part: {part_job_name}")
            self.manager.start(part, job_name=part_job_name)
            print("Started batch job. Checking status...")
            while True:
                status = self.manager.check_status(job_name=part_job_name)
                print(f"Status: {status}")
                if status == "completed":
                    print("Job completed. Fetching results...")
                    output_data, log = self.manager.fetch_results(
                        job_name=part_job_name, remove_cache=False
                    )
                    output_data = self.config.import_function(output_data)
                    self._save_results(output_data, log, part_idx)
                    print("Fetched and saved results for this part.")
                    return
                elif status == "failed":
                    print("Job failed. Clearing state, waiting, and retrying...")
                    self.manager._clear_state(part_job_name)
                    time.sleep(10)  # Wait before retrying
                    break  # Break inner loop to restart the job
                else:
                    time.sleep(5)  # Wait before checking again

    def _save_results(
        self, output_data: list[dict[str, Any]], log: list[Any], part_idx: int
    ):
        part_suffix = f"_part_{part_idx + 1}" if len(self.parts) > 1 else ""
        result_path = (
            Path(self.config.BASE_OUTPUT_DIR)
            / f"{Path(self.output_data_filename).stem}{part_suffix}.json"
        )
        if not output_data:
            print("No output data to save. Skipping this part.")
            return
        else:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
        if log:
            log_path = (
                Path(self.config.BASE_OUTPUT_DIR)
                / f"{Path(self.output_data_filename).stem}{part_suffix}_log.json"
            )
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log, f, ensure_ascii=False, indent=4)

    def _result_exists(self, part_idx: int) -> bool:
        part_suffix = f"_part_{part_idx + 1}" if len(self.parts) > 1 else ""
        result_path = (
            Path(self.config.BASE_OUTPUT_DIR)
            / f"{Path(self.output_data_path).stem}{part_suffix}.json"
        )
        return result_path.exists()


if __name__ == "__main__":
    print("=== Batch Job Runner ===")
    config = BatchConfig(
        system_prompt="",
        job_name="job_name",
        input_data_path="Data.json",
        output_data_filename="output",
    )
    runner = BatchJobRunner(config)
    runner.run()
