# scorer.py

import json
import logging
from typing import Any, Dict, List, Optional
from openai import OpenAI
from texttools.base.base_task_performer import BaseTaskPerformer


class GemmaScorer(BaseTaskPerformer):
    """
    A scorer component utilizing Gemma-style LLMs to evaluate the relevance of
    individual text results against a given query. It assigns a score from 0-5.
    Can optionally include a reasoning step for each result to enhance accuracy.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        temperature: float = 0.0,
        prompt_template: Optional[str] = None,
        use_reason: bool = False,
        handlers: Optional[List[Any]] = None,
        **client_kwargs: Any,
    ):
        """
        Initializes the GemmaScorer.

        :param client: An initialized OpenAI client (or compatible).
        :param model: The name of the LLM model to use for scoring (e.g., "gemma-7b-it").
        :param temperature: The sampling temperature for LLM generation (0.0 for deterministic).
        :param prompt_template: An optional initial system-level prompt for the LLM.
        :param use_reason: If True, the scorer will perform an internal reasoning step
                           for each result and include it in the scoring prompt.
        :param handlers: Optional list of handlers for dispatching scoring results.
        :param client_kwargs: Additional keyword arguments for the OpenAI client.
        """
        super().__init__(handlers)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs
        self.prompt_template = prompt_template
        self.use_reason = use_reason

        # Defines the expected JSON schema for the LLM's score output.
        self.score_schema = {"score": "integer"}

    def _build_messages(
        self, query: str, result_text: str, reason: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Constructs the messages payload for the LLM API call to score a single result.
        Now includes an optional 'reason' parameter.

        :param query: The search query.
        :param result_text: The text content of the result to be scored.
        :param reason: An optional reasoning summary generated internally for this specific result.
        :return: A list of message dictionaries formatted for the LLM API.
        """
        clean_query = self._preprocess(query)
        clean_result_text = self._preprocess(result_text)

        messages: List[Dict[str, str]] = []

        if self.prompt_template:
            messages.append({"role": "user", "content": self.prompt_template})

        # Conditionally add the reason to the prompt
        if self.use_reason and reason:
            messages.append(
                {
                    "role": "user",
                    "content": f"Consider this preliminary analysis for scoring: {reason}",
                }
            )

        messages.append({"role": "user", "content": f"Query: {clean_query}"})
        messages.append(
            {"role": "user", "content": f"Result text to score: {clean_result_text}"}
        )

        scoring_instruction = (
            "Score this result from 0 (not relevant) to 5 (highly relevant) based on how well it matches the query. "
            "Return only the score as a JSON object with a 'score' key."
        )
        messages.append({"role": "user", "content": scoring_instruction})

        schema_instr = f"Respond only in JSON format: {json.dumps(self.score_schema)}"
        messages.append({"role": "user", "content": schema_instr})

        messages.append({"role": "assistant", "content": "{"})
        return messages

    def _reason(self, query: str, result_text: str) -> str:
        """
        Generates a brief reasoning summary for a single result's relevance to the query.
        This summary is intended to provide additional context to the LLM for scoring.

        :param query: The search query.
        :param result_text: The text content of the result being analyzed.
        :return: A string containing the reasoning summary.
        """
        clean_query = self._preprocess(query)
        clean_result_text = self._preprocess(result_text)

        # Improved handling for result text truncation
        display_result_text = clean_result_text
        if len(clean_result_text) > 200:
            display_result_text = clean_result_text[:200] + "..."

        reason_prompt = f"""
            Analyze the relevance of the following result text to the given query.
            Focus on key terms, concepts, and overall intent.
            Query: "{clean_query}"
            Result Text: "{display_result_text}"
            
            Provide a brief summary of why this result might or might not be relevant to the query.
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

    def perform(self, query: str, result_text: str) -> int:
        """
        Scores a single result's text against a given query using the configured LLM.
        This is the main public method for the scorer.

        :param query: The search query string.
        :param result_text: The text content of the result to be scored.
        :return: An integer score (0-5) representing relevance.
        :raises ValueError: If result_text is empty, LLM output is malformed, or score is invalid.
        """
        if not result_text:
            logging.warning("Received empty result text for scoring.")
            self._dispatch(
                {
                    "query": query,
                    "result_text": result_text,
                    "score": 0,
                    "error": "Empty result text",
                }
            )
            return 0

        # Generate reason if enabled
        reason_for_scoring = None
        if self.use_reason:
            reason_for_scoring = self._reason(query, result_text)

        messages = self._build_messages(query, result_text, reason_for_scoring)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        raw_score_output = resp.choices[0].message.content.strip()

        # Robustly extract JSON if LLM adds preamble
        if not raw_score_output.startswith("{"):
            raw_score_output = "{" + raw_score_output
        try:
            parsed_score = json.loads(raw_score_output)
        except json.JSONDecodeError as e:
            logging.error(
                f"Failed to parse JSON for single result scoring: {e}\nRaw output: {raw_score_output}"
            )
            self._dispatch(
                {
                    "query": query,
                    "result_text": result_text,
                    "score": 0,
                    "error": f"JSON parsing failed: {e}",
                }
            )
            return 0

        score = parsed_score.get("score")
        if not isinstance(score, (int, float)) or not (0 <= score <= 5):
            logging.warning(
                f"LLM returned invalid score for result text '{result_text[:50]}...': {score}. "
                "Expected integer 0-5. Raw: {raw_score_output}"
            )
            self._dispatch(
                {
                    "query": query,
                    "result_text": result_text,
                    "score": 0,
                    "warning": "Invalid score format",
                }
            )
            return 0

        final_score = int(score)
        self._dispatch(
            {
                "query": query,
                "result_text": result_text,
                "score": final_score,
                "reason": reason_for_scoring,
            }
        )
        return final_score
