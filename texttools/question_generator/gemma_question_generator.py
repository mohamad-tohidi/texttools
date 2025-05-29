from typing import Any, Dict, List, Optional
import json
from openai import OpenAI
from texttools.base.base_question_generator import BaseQuestionGenerator


class GemmaQuestionGenerator(BaseQuestionGenerator):
    """
    Question Generator for Gemma-style models with optional reasoning step.
    Outputs JSON with a single string field: {"generated_question": "..."}.

    Allows optional extra instructions via `prompt_template`.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        use_reason: bool = False,
        temperature: float = 0.0,
        prompt_template: Optional[str] = None,
        handlers: Optional[List[Any]] = None,
        **client_kwargs: Any,
    ):
        super().__init__(handlers)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs

        self.use_reason = use_reason
        self.prompt_template = prompt_template

        # Define the JSON schema for the generated question output
        self.json_schema = {"generated_question": "string"}

    def _build_messages(
        self, answer: str, reason: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Builds the message list for the LLM API call for question generation.
        """
        clean_answer = self.preprocess(answer)
        messages: List[Dict[str, str]] = []

        if self.prompt_template:
            messages.append({"role": "user", "content": self.prompt_template})

        if reason:
            messages.append(
                {
                    "role": "user",
                    "content": f"Based on this analysis of the answer: {reason}",
                }
            )

        messages.append(
            {
                "role": "user",
                "content": "Given the following answer, generate a single, appropriate question that this answer would directly respond to.",
            }
        )
        messages.append({"role": "user", "content": clean_answer})

        # Ensure the schema is dumped as a valid JSON string for the LLM
        schema_instr = f"Respond only in JSON format: {json.dumps(self.json_schema)}"
        messages.append({"role": "user", "content": schema_instr})

        messages.append(
            {"role": "assistant", "content": "{"}
        )  # Hint to start JSON output
        return messages

    def _reason(self, answer: str) -> str:
        """
        Internal reasoning step to help the model understand the core information
        and implications of the answer.
        """
        messages = [
            {
                "role": "user",
                "content": """
                    Analyze the following answer to identify its key facts, main subject, and what kind of information it provides.
                    Provide a brief, summarized understanding of the answer's content that will help in formulating a relevant and direct question.
                    """,
            },
            {
                "role": "user",
                "content": f"""
                    {answer}
                    """,
            },
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )

        reason_summary = resp.choices[0].message.content.strip()
        return reason_summary

    def generate_question(self, answer: str) -> str:
        """
        Generates a question for the input `answer`.
        Optionally uses an internal reasoning step for better accuracy.
        """
        reason_summary = None
        if self.use_reason:
            reason_summary = self._reason(answer)

        messages = self._build_messages(answer, reason_summary)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        raw = resp.choices[0].message.content.strip()

        if not raw.startswith("{"):
            raw = "{" + raw
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON for question generation: {e}\nRaw output: {raw}"
            )

        generated_question = parsed.get("generated_question")

        if not isinstance(generated_question, str):
            raise ValueError(
                f"Invalid response schema for question generation. Expected 'generated_question' as a string, got: {parsed}"
            )

        # dispatch and return
        self._dispatch(
            {
                "original_answer": answer,
                "generated_question": generated_question,
            }
        )
        return generated_question
