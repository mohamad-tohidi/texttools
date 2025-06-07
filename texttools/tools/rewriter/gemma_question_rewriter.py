from typing import Any, Dict, List, Optional
import json
from openai import OpenAI
from texttools.base.base_question_rewriter import BaseQuestionRewriter, RewriteMode
from texttools.formatter import Gemma3Formatter
from pydantic import BaseModel

# class QuestionGeneration(BaseModel):
#     generated_question: str


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
        chat_formatter: Optional[Any] = None,
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
        
        self.chat_formatter = chat_formatter or Gemma3Formatter()
        

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
                "**respond in the language of the question**"
            )
        elif mode == RewriteMode.DIFFERENT_MEANING_SIMILAR_WORDING:
            instruction = (
                "Rewrite the following question using *very similar wording and phrasing* "
                "to the original, but ensure the rewritten question has a *completely different meaning*. "
                "Focus on subtle changes that drastically alter the intent or subject of the question."
                "**respond in the language of the question**"
            )
        else:
            raise ValueError(f"Unsupported rewrite mode: {mode}")

        messages.append({"role": "user", "content": instruction})
        messages.append({"role": "user", "content": f"here is the question: {clean_question}"})

        # schema_instr = f"Respond only in JSON format: {json.dumps(self.json_schema)}"
        schema_instr = f"Respond only with the generated question, without any additional information."
        messages.append({"role": "user", "content": schema_instr})

        # messages.append({"role": "assistant", "content": "{"})
        # deprecated method for structured output
        
        
        # this line will restructure the messages
        # based on the formatter that we provided
        # some models will require custom settings
        restructured = self.chat_formatter.format(messages=messages)
        
        return restructured

    def _reason(self, question: str, mode: RewriteMode) -> str:
        """
        Internal reasoning step to help the model understand the core meaning
        or structure of the question depending on the mode.
        """
        if mode == RewriteMode.SAME_MEANING_DIFFERENT_WORDING:
            reason_prompt = """
                Analyze the following question to identify its core intent, key concepts, 
                and the specific information it is seeking.
                Provide a brief, summarized understanding of the question's meaning that
                will help in rephrasing it accurately without changing its intent.
                
                **respond in the language of the question**
                
                """
        elif mode == RewriteMode.DIFFERENT_MEANING_SIMILAR_WORDING:
            reason_prompt = """
                Analyze the following question to identify its exact wording, phrasing,
                and the literal meaning it conveys.
                Provide a brief, summarized analysis of its linguistic structure and current meaning,
                which will then be used to create a new question with similar words but a different meaning.
                
                **respond in the language of the question**
                """
        else:
            raise ValueError(f"Unsupported rewrite mode for reason: {mode}")

        messages = [
            {"role": "user", "content": reason_prompt},
            {"role": "user", "content": f"here is the question: {question}"},
        ]
        
        restructured = self.chat_formatter.format(messages=messages)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=restructured,
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

        # for structured output formatting
        # but now i want to try somthing else
        # i want to see if i could get the results without structured output        
        # completion = self.client.beta.chat.completions.parse(
        #     model=self.model,
        #     messages=messages,
        #     response_format=QuestionGeneration,
        #     temperature=self.temperature,
        #     extra_body=dict(guided_decoding_backend="outlines"),
        #     **self.client_kwargs,
        # )
        # message = completion.choices[0].message
        # if message.parsed:
        #     result = message.parsed.generated_question
        # else:
        #     raise ValueError(f"Failed to parse the response. Raw content: {message.content}")
        

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )

        result = resp.choices[0].message.content.strip()


        
        
        # dispatch and return
        self._dispatch(
            {
                "original_question": question,
                "rewritten_question": result,
                "mode": mode.value,  
            }
        )
        return result
