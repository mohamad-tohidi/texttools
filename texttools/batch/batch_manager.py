import json
import uuid
from pathlib import Path
from typing import Type, TypeVar
import logging

from pydantic import BaseModel
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema

# Base Model type for output models
T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("texttools.batch_manager")


class BatchManager:
    """
    Manages batch processing jobs for OpenAI's chat completions with structured outputs.

    Handles the full lifecycle of a batch job: creating tasks from input texts,
    starting the job, monitoring status, and fetching results. Results are automatically
    parsed into the specified Pydantic output model. Job state is persisted to disk.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        output_model: Type[T],
        prompt_template: str,
        state_dir: Path = Path(".batch_jobs"),
        custom_json_schema_obj_str: dict | None = None,
        **client_kwargs: object,
    ):
        self._client = client
        self._model = model
        self._output_model = output_model
        self._prompt_template = prompt_template
        self._state_dir = state_dir
        self._custom_json_schema_obj_str = custom_json_schema_obj_str
        self._client_kwargs = client_kwargs
        self._dict_input = False
        self._state_dir.mkdir(parents=True, exist_ok=True)

        if custom_json_schema_obj_str and not isinstance(
            custom_json_schema_obj_str, dict
        ):
            raise ValueError("Schema should be a dict")

    def _state_file(self, job_name: str) -> Path:
        return self._state_dir / f"{job_name}.json"

    def _load_state(self, job_name: str) -> list[dict[str, object]]:
        """
        Loads the state (job information) from the state file for the given job name.
        Returns an empty list if the state file does not exist.
        """
        path = self._state_file(job_name)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_state(self, job_name: str, jobs: list[dict[str, object]]) -> None:
        """
        Saves the job state to the state file for the given job name.
        """
        with open(self._state_file(job_name), "w", encoding="utf-8") as f:
            json.dump(jobs, f)

    def _clear_state(self, job_name: str) -> None:
        """
        Deletes the state file for the given job name if it exists.
        """
        path = self._state_file(job_name)
        if path.exists():
            path.unlink()

    def _build_task(self, text: str, idx: str) -> dict[str, object]:
        """
        Builds a single task dictionary for the batch job, including the prompt, model, and response format configuration.
        """
        response_format_config: dict[str, object]

        if self._custom_json_schema_obj_str:
            response_format_config = {
                "type": "json_schema",
                "json_schema": self._custom_json_schema_obj_str,
            }
        else:
            raw_schema = to_strict_json_schema(self._output_model)
            response_format_config = {
                "type": "json_schema",
                "json_schema": {
                    "name": self._output_model.__name__,
                    "schema": raw_schema,
                },
            }

        return {
            "custom_id": str(idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self._prompt_template},
                    {"role": "user", "content": text},
                ],
                "response_format": response_format_config,
                **self._client_kwargs,
            },
        }

    def _prepare_file(self, payload: list[str] | list[dict[str, str]]) -> Path:
        """
        Prepares a JSONL file containing all tasks for the batch job, based on the input payload.
        Returns the path to the created file.
        """
        if not payload:
            raise ValueError("Payload must not be empty")
        if isinstance(payload[0], str):
            tasks = [self._build_task(text, uuid.uuid4().hex) for text in payload]
        elif isinstance(payload[0], dict):
            tasks = [self._build_task(dic["text"], dic["id"]) for dic in payload]

        else:
            raise TypeError(
                "The input must be either a list of texts or a dictionary in the form {'id': str, 'text': str}"
            )

        file_path = self._state_dir / f"batch_{uuid.uuid4().hex}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
        return file_path

    def start(self, payload: list[str | dict[str, str]], job_name: str):
        """
        Starts a new batch job by uploading the prepared file and creating a batch job on the server.
        If a job with the same name already exists, it does nothing.
        """
        if self._load_state(job_name):
            return

        path = self._prepare_file(payload)
        upload = self._client.files.create(file=open(path, "rb"), purpose="batch")
        job = self._client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        ).to_dict()
        self._save_state(job_name, [job])

    def check_status(self, job_name: str) -> str:
        """
        Checks and returns the current status of the batch job with the given job name.
        Updates the job state with the latest information from the server.
        """
        job = self._load_state(job_name)[0]
        if not job:
            return "completed"

        info = self._client.batches.retrieve(job["id"])
        job = info.to_dict()
        self._save_state(job_name, [job])
        logger.info("Batch job status: %s", job)
        return job["status"]

    def fetch_results(
        self, job_name: str, remove_cache: bool = True
    ) -> tuple[dict[str, str], list]:
        """
        Fetches the results of a completed batch job. Optionally saves the results to a file and/or removes the job cache.
        Returns a tuple containing the parsed results and a log of errors (if any).
        """
        job = self._load_state(job_name)[0]
        if not job:
            return {}
        batch_id = job["id"]

        info = self._client.batches.retrieve(batch_id)
        out_file_id = info.output_file_id
        if not out_file_id:
            error_file_id = info.error_file_id
            if error_file_id:
                err_content = (
                    self._client.files.content(error_file_id).read().decode("utf-8")
                )
                logger.error("Error file content:", err_content)
            return {}

        content = self._client.files.content(out_file_id).read().decode("utf-8")
        lines = content.splitlines()
        results = {}
        log = []
        for line in lines:
            result = json.loads(line)
            custom_id = result["custom_id"]
            if result["response"]["status_code"] == 200:
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                try:
                    parsed_content = json.loads(content)
                    model_instance = self._output_model(**parsed_content)
                    results[custom_id] = model_instance.model_dump(mode="json")
                except json.JSONDecodeError:
                    results[custom_id] = {"error": "Failed to parse content as JSON"}
                    error_d = {custom_id: results[custom_id]}
                    log.append(error_d)
                except Exception as e:
                    results[custom_id] = {"error": str(e)}
                    error_d = {custom_id: results[custom_id]}
                    log.append(error_d)
            else:
                error_message = (
                    result["response"]["body"]
                    .get("error", {})
                    .get("message", "Unknown error")
                )
                results[custom_id] = {"error": error_message}
                error_d = {custom_id: results[custom_id]}
                log.append(error_d)

        if remove_cache:
            self._clear_state(job_name)

        return results, log
