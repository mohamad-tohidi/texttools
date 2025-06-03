from typing import Any, Dict, List
from pydantic import BaseModel, create_model
from openai import OpenAI
from texttools.base.base_question_detector import BaseQuestionDetector

class LLMQuestionDetector(BaseQuestionDetector):
    """
    LLM-based binary question detector that wraps OpenAI s structured output parsing.

    Usage:
        ```python
        from openai import OpenAI
        from texttools import LLMQuestionDetector

        # Instantiate an OpenAI client (ensure you ve set OPENAI_API_KEY)
        client = OpenAI()

        # Create detector
        detector = LLMQuestionDetector(
            client=client,
            model="gpt-4o-2024-08-06",
            temperature=0.0,  # deterministic outputs
            prompt_template=(
                "You are a binary classifier. "
                "Answer only with `true` or `false` depending on the input."
            ),
            handlers=[my_handler],  # optional callbacks on each detection
            max_tokens=10           # any other OpenAIClient kwargs
        )

        # Detect whether a string is a question
        is_question = detector.detect("How are you today?")
        # is_question == True
        ```
        
    Parameters:
        client (OpenAI):
            Instantiated OpenAI client. Make sure your API key is configured.
        model (str):
            Model name to use (e.g. "gpt-4", "gpt-4o-2024-08-06").
        temperature (float, default=0.0):
            Sampling temperature; 0.0 yields deterministic outputs.
        prompt_template (str, optional):
            System‐level instructions guiding the classification.
        handlers (List[callable], optional):
            List of callables that receive {"text": bool} after each detect().
        client_kwargs (Any):
            Additional parameters passed directly to OpenAI (e.g., max_tokens, top_p).

    Internals:
        - Wraps your input in system/user messages.
        - Uses Pydantic to enforce that the API returns a boolean.
        - Dispatches result to any registered handlers.

    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        temperature: float = 0.0,
        prompt_template: str = None,
        handlers: List[Any] = None,
        **client_kwargs: Any
    ):
        """
        :param client: an instantiated OpenAI client
        :param model: the model name (e.g. "gpt-4o-2024-08-06")
        :param temperature: sampling temperature
        :param prompt_template: override default prompt instructions
        :param handlers: optional list of result handlers
        :param client_kwargs: any other OpenAI kwargs (e.g. `max_tokens`, `top_p`, etc.)
        """
        super().__init__(handlers)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs

        self.prompt_template = prompt_template or (
            "You are a binary classifier. "
            "Answer only with `true` or `false` depending on the input."
        )

        self._OutputModel = create_model(
            "DetectionOutput",
            result=(bool, ...),  
        )

    def _build_messages(self, text: str) -> List[Dict[str, str]]:
        clean = self.preprocess(text)
        return [
            {"role": "system", "content": self.prompt_template},
            {"role": "user",   "content": clean},
        ]

    def detect(self, text: str) -> bool:
        msgs = self._build_messages(text)
        resp = self.client.responses.parse(
            model=self.model,
            input=msgs,
            text_format=self._OutputModel,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        output: BaseModel = resp.output_parsed
        self._dispatch({
            "question": text,             
            "result":   output.result     
        })
        return output.result
