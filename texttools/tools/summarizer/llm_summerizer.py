from typing import Any, Dict, List, Optional
from pydantic import BaseModel, create_model
from openai import OpenAI
from texttools.base.base_summarizer import BaseSummarizer


class LLMSummarizer(BaseSummarizer):
    """
    LLM-based text summarizer that wraps OpenAI's structured output parsing.

    Usage:
        ```python
        from openai import OpenAI
        from texttools import LLMSummarizer

        client = OpenAI()
        summarizer = LLMSummarizer(
            client=client,
            model="gpt-4o-2024-08-06",
            temperature=0.7,
            prompt_template=(
                "You are a helpful assistant that produces concise summaries of the provided text."
            ),
            handlers=[my_handler],  # optional callbacks on each summarization
            max_tokens=150,         # any other OpenAIClient kwargs
        )

        summary = summarizer.summarize("Long article text...")
        print(summary)
        ```

    Parameters:
        client (OpenAI):
            Instantiated OpenAI client. Ensure your API key is configured.
        model (str):
            Model name to use (e.g., "gpt-4").
        temperature (float, default=0.7):
            Sampling temperature.
        prompt_template (str, optional):
            System-level instructions guiding the summarization.
        handlers (List[callable], optional):
            List of callables that receive {"summary": str, "original_text": str}.
        client_kwargs (Any):
            Additional parameters passed directly to OpenAI (e.g., max_tokens, top_p).
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        temperature: float = 0.7,
        prompt_template: Optional[str] = None,
        handlers: Optional[List[Any]] = None,
        **client_kwargs: Any
    ):
        super().__init__(handlers)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs

        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else (
                """
                You are a helpful assistant that produces concise and accurate summaries of the provided text.
                do not explain anything, onlu provide the summarized version.
                """
            )
        )

        self._OutputModel = create_model(
            "SummarizationOutput",
            summary=(str, ...),
        )

    def _build_messages(self, text: str) -> List[Dict[str, str]]:
        preprocessed = self.preprocess(text)
        return [
            {"role": "system", "content": self.prompt_template},
            {"role": "user", "content": preprocessed},
        ]

    def summarize(self, text: str) -> str:
        """
        Generate a summary for the input text.

        :param text: The text to summarize.
        :return: A summary string.
        """
        messages = self._build_messages(text)
        resp = self.client.responses.parse(
            model=self.model,
            input=messages,
            text_format=self._OutputModel,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        output: BaseModel = resp.output_parsed
        summary_text: str = output.summary

        self._dispatch(summary=summary_text, original_text=text)

        return summary_text
