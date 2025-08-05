import json
import logging
from typing import Any, Optional

from openai import OpenAI

from texttools.base.base_task_performer import BaseTaskPerformer


class GemmaSorter(BaseTaskPerformer):
    """
    A sorter component utilizing Gemma-style LLMs to order a list of
    pre-scored results based on a query, handling ties semantically.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        temperature: float = 0.0,
        prompt_template: Optional[str] = None,
        use_reason: bool = False,
        handlers: Optional[list[Any]] = None,
        **client_kwargs: Any,
    ):
        """
        Initializes the GemmaSorter.

        :param client: An initialized OpenAI client (or compatible).
        :param model: The name of the LLM model to use for sorting (e.g., "gemma-7b-it").
        :param temperature: The sampling temperature for LLM generation (0.0 for deterministic).
        :param prompt_template: An optional initial system-level prompt for the LLM.
        :param use_reason: If True, the sorter will perform an internal reasoning step
                           and include it in the sorting prompt.
        :param handlers: Optional list of handlers for dispatching sorting results.
        :param client_kwargs: Additional keyword arguments for the OpenAI client.
        """
        super().__init__(handlers)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs
        self.prompt_template = prompt_template
        self.use_reason = use_reason

        # Defines the expected JSON schema for the LLM's ordered IDs output.
        self.sort_schema = {"ordered_ids": ["string"]}

    def _build_sorting_messages(
        self,
        query: str,
        scored_results: list[dict[str, Any]],
        reason: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """
        Constructs the messages payload for the LLM API call to sort results.

        :param query: The original search query.
        :param scored_results: A list of dictionaries, where each dict has '_internal_id', 'text', and 'score'.
        :param reason: An optional reasoning summary to provide context to the LLM.
        :return: A list of message dictionaries formatted for the LLM API.
        """
        clean_query = self._preprocess(query)
        messages: list[dict[str, str]] = []

        if self.prompt_template:
            messages.append({"role": "user", "content": self.prompt_template})

        if self.use_reason and reason:
            messages.append(
                {
                    "role": "user",
                    "content": f"Based on this analysis: {reason}",
                }
            )

        messages.append({"role": "user", "content": f"Original Query: {clean_query}"})

        # Format the scored results for the LLM.
        scored_results_presentation = []
        for res_dict in scored_results:
            scored_results_presentation.append(
                f"ID: {res_dict['_internal_id']}\nScore: {res_dict.get('score', 'N/A')}\nText: {res_dict.get('text', 'N/A')}"
            )
        messages.append(
            {
                "role": "user",
                "content": "Here are the results with their assigned scores:\n"
                + "\n---\n".join(scored_results_presentation),
            }
        )

        sorting_instruction = (
            "Based on the provided scores, sort the result 'id's into an ordered list from most relevant to least relevant. "
            "If multiple results have the same score, use semantic similarity to the original query as a tie-breaker. "
            "Return only a JSON list of the 'id's in the final sorted order."
        )
        messages.append({"role": "user", "content": sorting_instruction})

        schema_instr = f"Respond only in JSON format: {json.dumps(self.sort_schema)}"
        messages.append({"role": "user", "content": schema_instr})
        messages.append({"role": "assistant", "content": "{"})
        return messages

    def _reason(self, query: str, results: list[dict[str, Any]]) -> str:
        """
        Generates an internal reasoning summary to help the LLM with sorting,
        especially for tie-breaking. This summary is based on the query and initial results.

        :param query: The original search query.
        :param results: A list of results, potentially including scores and IDs.
        :return: A string containing the reasoning summary.
        """
        clean_query = self._preprocess(query)

        # Truncate results for reasoning prompt to avoid exceeding token limits
        results_for_reasoning_display = []
        for res in results:
            text_snippet = res.get("text", "")
            if len(text_snippet) > 100:
                text_snippet = text_snippet[:100] + "..."
            results_for_reasoning_display.append(text_snippet)

        reason_prompt = f"""
            Analyze the original query: "{clean_query}"
            And consider these initial result snippets (with their scores if available): {results_for_reasoning_display}

            Formulate a brief analysis focusing on how to best order these results, especially considering tie-breaking rules based on semantic similarity to the query.
            """

        messages = [
            {"role": "user", "content": reason_prompt},
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        return resp.choices[0].message.content.strip()

    def perform(
        self, query: str, scored_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Sorts a list of results (each with an assigned score and an '_internal_id')
        based on the query using the configured LLM. This is the main public method for the sorter.

        :param query: The original search query string.
        :param scored_results: A list of dictionaries, where each dict has at least
                               '_internal_id', 'text', and 'score' keys.
        :return: A list of result dictionaries representing the final sorted order.
                 Each dictionary will have the '_internal_id' removed.
        :raises ValueError: If scored_results is empty, LLM output is malformed, or IDs are invalid.
        """
        if not scored_results:
            logging.info("Received empty list of scored results for sorting.")
            self._dispatch(
                {"query": query, "scored_results": scored_results, "ordered_ids": []}
            )
            return []

        # Prepare a map for quick lookup of full result objects by their internal ID
        id_to_full_result_map: dict[str, dict[str, Any]] = {}
        for i, res in enumerate(scored_results):
            if "_internal_id" not in res:
                # Assign a temporary internal ID if missing, important for LLM interaction
                res["_internal_id"] = res.get("id", f"gen_id_{i}")
            if "score" not in res:
                logging.warning(
                    f"Result ID '{res['_internal_id']}' missing score for sorting. Defaulting to 0."
                )
                res["score"] = 0
            id_to_full_result_map[res["_internal_id"]] = res.copy()  # Store a copy

        reason_summary = None
        if self.use_reason:
            reason_summary = self._reason(query, list(id_to_full_result_map.values()))

        messages_sort = self._build_sorting_messages(
            query, list(id_to_full_result_map.values()), reason_summary
        )

        resp_sort = self.client.chat.completions.create(
            model=self.model,
            messages=messages_sort,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        raw_sort_output = resp_sort.choices[0].message.content.strip()

        # Robustly extract JSON if LLM adds preamble
        if not raw_sort_output.startswith("{"):
            raw_sort_output = "{" + raw_sort_output

        try:
            parsed_sort = json.loads(raw_sort_output)
        except json.JSONDecodeError as e:
            logging.error(
                f"Failed to parse JSON for sorting step: {e}\nRaw output: {raw_sort_output}"
            )
            self._dispatch(
                {
                    "query": query,
                    "scored_results": scored_results,
                    "ordered_ids_from_llm": [],
                    "final_sorted_results": [],
                    "error": f"JSON parsing failed: {e}",
                }
            )
            raise ValueError(
                f"LLM output is not valid JSON for sorting: {raw_sort_output}"
            ) from e

        llm_ordered_ids = parsed_sort.get("ordered_ids")
        if not isinstance(llm_ordered_ids, list) or not all(
            isinstance(item, str) for item in llm_ordered_ids
        ):
            logging.warning(
                f"LLM returned invalid sort schema. Expected 'ordered_ids' as a list of strings, got: {parsed_sort}"
            )
            self._dispatch(
                {
                    "query": query,
                    "scored_results": scored_results,
                    "ordered_ids_from_llm": llm_ordered_ids,
                    "final_sorted_results": [],
                    "warning": "Invalid sort format",
                }
            )
            raise ValueError(
                f"LLM returned invalid 'ordered_ids' format: {llm_ordered_ids}"
            )

        final_sorted_results: list[dict[str, Any]] = []
        ids_placed_by_llm_set = set()

        for internal_id in llm_ordered_ids:
            if (
                internal_id in id_to_full_result_map
                and internal_id not in ids_placed_by_llm_set
            ):
                result_to_add = id_to_full_result_map[internal_id].copy()
                result_to_add.pop("_internal_id", None)
                final_sorted_results.append(result_to_add)
                ids_placed_by_llm_set.add(internal_id)
            elif internal_id not in id_to_full_result_map:
                logging.warning(
                    f"LLM ordered ID '{internal_id}' not found in original results. Skipping."
                )

        unranked_results = []
        for res in scored_results:
            if res["_internal_id"] not in ids_placed_by_llm_set:
                result_to_add = res.copy()
                result_to_add.pop("_internal_id", None)
                unranked_results.append(result_to_add)

        if unranked_results:  # Add warning here
            logging.warning(
                f"The LLM did not explicitly rank {len(unranked_results)} result(s). "
                "These will be appended to the end of the sorted list, ordered by their original score."
            )

        unranked_results.sort(key=lambda x: x.get("score", -1), reverse=True)
        final_sorted_results.extend(unranked_results)

        self._dispatch(
            {
                "query": query,
                "original_scored_results": scored_results,
                "llm_ordered_ids": llm_ordered_ids,
                "final_sorted_results": final_sorted_results,
            }
        )
        return final_sorted_results
