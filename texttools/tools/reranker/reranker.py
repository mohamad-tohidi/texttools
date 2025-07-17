# reranker.py

import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from texttools.base.base_task_performer import BaseTaskPerformer

from .scorer import GemmaScorer
from .sorter import GemmaSorter


class GemmaReranker(BaseTaskPerformer):
    """
    A Reranker component that orchestrates a GemmaScorer and GemmaSorter
    to refine the order of a list of search results based on a query.
    It first scores individual results and then sorts them.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        scorer_prompt_template: Optional[str] = None,
        sorter_prompt_template: Optional[str] = None,
        scorer_use_reason: bool = False,
        sorter_use_reason: bool = False,
        temperature: float = 0.0,
        handlers: Optional[List[Any]] = None,
        **client_kwargs: Any,
    ):
        """
        Initializes the GemmaReranker with configurations for its internal scorer and sorter.

        :param client: An initialized OpenAI client (or compatible).
        :param model: The LLM model to use for both scoring and sorting (e.g., "gemma-7b-it").
        :param scorer_prompt_template: Optional initial system-level prompt for the scorer.
        :param sorter_prompt_template: Optional initial system-level prompt for the sorter.
        :param scorer_use_reason: If True, the internal scorer will use an internal reasoning step.
        :param sorter_use_reason: If True, the internal sorter will use an internal reasoning step.
        :param temperature: The sampling temperature for LLM generation (0.0 for deterministic).
        :param handlers: Optional list of handlers for dispatching reranking results.
        :param client_kwargs: Additional keyword arguments for the OpenAI client.
        """
        super().__init__(handlers)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs

        # Initialize the internal Scorer and Sorter components
        self.scorer = GemmaScorer(
            client=self.client,
            model=self.model,
            temperature=self.temperature,
            prompt_template=scorer_prompt_template,
            use_reason=scorer_use_reason,
            handlers=[],
            **self.client_kwargs,
        )
        self.sorter = GemmaSorter(
            client=self.client,
            model=self.model,
            temperature=self.temperature,
            prompt_template=sorter_prompt_template,
            use_reason=sorter_use_reason,
            handlers=[],
            **self.client_kwargs,
        )

    def perform(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Performs the complete reranking operation: scoring each result and then sorting them.

        :param query: The original search query.
        :param results: A list of result dictionaries to be reranked.
                        Each dictionary should ideally have a 'text' key.
        :return: A list of result dictionaries, reranked by relevance.
                 Each dictionary will have an added 'score' key.
        :raises ValueError: If the input results list is empty or if sub-components fail.
        """
        if not results:
            logging.info(
                "Received empty list of results for reranking. Returning empty list."
            )
            self._dispatch(
                {"query": query, "original_results": results, "reranked_results": []}
            )
            return []

        scored_results: List[Dict[str, Any]] = []

        # Step 1: Score each individual result
        for i, res in enumerate(results):
            # Create a unique internal ID if not already present
            _internal_id = res.get("id", f"rerank_id_{i}")

            result_text = res.get("text")
            if not result_text:
                logging.warning(
                    f"Result with ID '{_internal_id}' has no 'text' key. Skipping scoring."
                )
                score = 0  # Default score if no text
            else:
                try:
                    score = self.scorer.perform(query, result_text)
                except Exception as e:
                    logging.error(
                        f"Scorer failed for result ID '{_internal_id}': {e}",
                        exc_info=True,
                    )
                    score = 0  # Default score on scorer failure

            # Create a new dictionary to add the score and internal ID
            # It's important to copy the original result and add these for the sorter.
            scored_res_copy = res.copy()
            scored_res_copy["score"] = score
            scored_res_copy["_internal_id"] = _internal_id  # Sorter expects this
            scored_results.append(scored_res_copy)

        # Step 2: Sort the scored results
        # The sorter's perform method expects a list of dictionaries with '_internal_id' and 'score'
        reranked_results = self.sorter.perform(query, scored_results)

        # Dispatch the final reranked results
        self._dispatch(
            {
                "query": query,
                "original_results": results,
                "scored_results": scored_results,  # Can include internal IDs for debugging if needed
                "reranked_results": reranked_results,  # This will have '_internal_id' removed by sorter
            }
        )

        return reranked_results
