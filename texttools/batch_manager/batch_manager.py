import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema

# from openai.lib._parsing._completions import type_to_response_format_param
from pydantic import BaseModel


class SimpleBatchManager:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        output_model: Type[BaseModel],
        prompt_template: str,
        handlers: Optional[List[Any]] = None,
        state_dir: Path = Path(".batch_jobs"),
        custom_json_schema_obj_str: Optional[dict] = None,
        **client_kwargs: Any,
    ):
        self.client = client
        self.model = model
        self.output_model = output_model
        self.prompt_template = prompt_template
        self.handlers = handlers or []
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.custom_json_schema_obj_str = custom_json_schema_obj_str
        self.client_kwargs = client_kwargs

        if self.custom_json_schema_obj_str is not dict:
            raise ValueError("schema should be a dict")

    def _state_file(self, job_name: str) -> Path:
        return self.state_dir / f"{job_name}.json"

    def _load_state(self, job_name: str) -> List[Dict[str, Any]]:
        path = self._state_file(job_name)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_state(self, job_name: str, jobs: List[Dict[str, Any]]):
        with open(self._state_file(job_name), "w", encoding="utf-8") as f:
            json.dump(jobs, f)

    def _clear_state(self, job_name: str):
        path = self._state_file(job_name)
        if path.exists():
            path.unlink()

    def _build_task(self, text: str, idx: int) -> Dict[str, Any]:
        response_format_config: Dict[str, Any]
        if self.custom_json_schema_obj_str:
            # try:
            # parsed_custom_schema = json.loads(self.custom_json_schema_obj_str)
            response_format_config = {
                "type": "json_schema",
                "json_schema": self.custom_json_schema_obj_str,
            }
        # except json.JSONDecodeError as e:
        #     raise ValueError(
        #         "Failed to parse custom_json_schema_obj_str. "
        #         "Please ensure it's a valid JSON string."
        #     ) from e
        else:
            raw_schema = to_strict_json_schema(self.output_model)
            response_format_config = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.output_model.__name__,
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
                    {"role": "system", "content": self.prompt_template},
                    {"role": "user", "content": text},
                ],
                "response_format": response_format_config,
                **self.client_kwargs,
            },
        }

    def _prepare_file(self, payload: List[str] | Dict[str, str]) -> Path:
        if isinstance(payload, list):
            tasks = [self._build_task(text, uuid.uuid4().hex) for text in payload]
        elif isinstance(payload, dict):
            tasks = [self._build_task(text, idx) for idx, text in payload.items()]
        else:
            raise TypeError(
                "The input must be either a list of texts or a dictionary in the form {'id': 'text'}."
            )

        file_path = self.state_dir / f"batch_{uuid.uuid4().hex}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
        return file_path

    def start(self, payload: List[str] | Dict[str, str], job_name: str):
        """
        The data is sent to the server as a single batch for processing.

        Args:
            payload (List[str] | Dict[str, str]): The data can be input either as a list of texts or as a dictionary in the form {'id': 'text'}. If a list of texts is provided without IDs, the IDs will be generated randomly.
            job_name (str): The job_name is used so that you can fetch the outputs after they are ready.
        """
        if self._load_state(job_name):
            return
        path = self._prepare_file(payload)
        upload = self.client.files.create(file=open(path, "rb"), purpose="batch")
        job = self.client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        ).to_dict()
        self._save_state(job_name, [job])

    def check_status(self, job_name: str) -> str:
        job = self._load_state(job_name)[0]
        if not job:
            return "completed"

        info = self.client.batches.retrieve(job["id"])
        job = info.to_dict()
        self._save_state(job_name, [job])
        print("HERE is the fucking job", job)
        return job["status"]

    def fetch_results(self, job_name: str) -> Dict[str, Any]:
        job = self._load_state(job_name)[0]
        if not job:
            return {}
        batch_id = job["id"]

        info = self.client.batches.retrieve(batch_id)
        out_file_id = info.output_file_id
        if not out_file_id:
            error_file_id = info.error_file_id
            if error_file_id:
                err_content = (
                    self.client.files.content(error_file_id).read().decode("utf-8")
                )
                print("Error file content:", err_content)
            return {}

        content = self.client.files.content(out_file_id).read().decode("utf-8")
        lines = content.splitlines()
        results = {}
        for line in lines:
            result = json.loads(line)
            custom_id = result["custom_id"]
            if result["response"]["status_code"] == 200:
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                try:
                    parsed_content = json.loads(content)
                    model_instance = self.output_model(**parsed_content)
                    results[custom_id] = model_instance
                except json.JSONDecodeError:
                    results[custom_id] = {"error": "Failed to parse content as JSON"}
                except Exception as e:
                    results[custom_id] = {"error": str(e)}
            else:
                error_message = (
                    result["response"]["body"]
                    .get("error", {})
                    .get("message", "Unknown error")
                )
                results[custom_id] = {"error": error_message}

        for handler in self.handlers:
            handler.handle(results)
        self._clear_state(job_name)
        return results
