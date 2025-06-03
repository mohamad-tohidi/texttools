from typing import Any, Dict, List, Optional
import json
from openai import OpenAI
from texttools.base.base_question_rewriter import BaseQuestionRewriter, RewriteMode


class GemmaQuestionRewriter(BaseQuestionRewriter):
    """
    Question Rewriter for Gemma-style models with two modes:
    1. Rewrite with same meaning, different wording.
    2. Rewrite with different meaning, similar wording.
    Outputs JSON with a single string field: {"rewritten_question": "..."}.

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

        self.json_schema = {"rewritten_question": "string"}

    def _build_messages(
        self, question: str, mode: RewriteMode, reason: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Builds the message list for the LLM API call for question rewriting,
        adapting the prompt based on the chosen mode.
        """
        clean_question = self.preprocess(question)
        messages: List[Dict[str, str]] = []

        if self.prompt_template:
            messages.append({"role": "user", "content": self.prompt_template})

        if reason:
            messages.append(
                {
                    "role": "user",
                    "content": f"Based on this analysis: {reason}",
                }
            )

        if mode == RewriteMode.SAME_MEANING_DIFFERENT_WORDING:
            instruction = (
                "Rewrite the following question using completely different wording and phrasing, "
                "ensuring its original meaning is perfectly preserved. The rewritten question "
                "should be distinct from the original but convey the exact same inquiry."
            )
        elif mode == RewriteMode.DIFFERENT_MEANING_SIMILAR_WORDING:
            instruction = (
                "Rewrite the following question using *very similar wording and phrasing* "
                "to the original, but ensure the rewritten question has a *completely different meaning*. "
                "Focus on subtle changes that drastically alter the intent or subject of the question."
            )
        else:
            raise ValueError(f"Unsupported rewrite mode: {mode}")

        messages.append({"role": "user", "content": instruction})
        messages.append({"role": "user", "content": clean_question})

        schema_instr = f"Respond only in JSON format: {json.dumps(self.json_schema)}"
        messages.append({"role": "user", "content": schema_instr})

        messages.append({"role": "assistant", "content": "{"})
        return messages

    def _reason(self, question: str, mode: RewriteMode) -> str:
        """
        Internal reasoning step to help the model understand the core meaning
        or structure of the question depending on the mode.
        """
        if mode == RewriteMode.SAME_MEANING_DIFFERENT_WORDING:
            reason_prompt = """
                Analyze the following question to identify its core intent, key concepts, and the specific information it is seeking.
                Provide a brief, summarized understanding of the question's meaning that will help in rephrasing it accurately without changing its intent.
                """
        elif mode == RewriteMode.DIFFERENT_MEANING_SIMILAR_WORDING:
            reason_prompt = """
                Analyze the following question to identify its exact wording, phrasing, and the literal meaning it conveys.
                Provide a brief, summarized analysis of its linguistic structure and current meaning, which will then be used to create a new question with similar words but a different meaning.
                """
        else:
            raise ValueError(f"Unsupported rewrite mode for reason: {mode}")

        messages = [
            {"role": "user", "content": reason_prompt},
            {"role": "user", "content": f"{question}"},
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )

        reason_summary = resp.choices[0].message.content.strip()
        return reason_summary

    def rewrite_question(
        self,
        question: str,
        mode: RewriteMode = RewriteMode.SAME_MEANING_DIFFERENT_WORDING,
    ) -> str:
        """
        Rewrites the input `question` based on the specified `mode`.
        Optionally uses an internal reasoning step for better accuracy.
        """
        reason_summary = None
        if self.use_reason:
            reason_summary = self._reason(question, mode)

        messages = self._build_messages(question, mode, reason_summary)
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
                f"Failed to parse JSON for question rewriting: {e}\nRaw output: {raw}"
            )

        rewritten_question = parsed.get("rewritten_question")

        if not isinstance(rewritten_question, str):
            raise ValueError(
                f"Invalid response schema for question rewriting. Expected 'rewritten_question' as a string, got: {parsed}"
            )

        # Dispatch includes the mode for logging/tracking purposes
        self._dispatch(
            {
                "original_question": question,
                "rewritten_question": rewritten_question,
                "mode": mode.value,  # Dispatch the string value of the Enum
            }
        )
        return rewritten_question
