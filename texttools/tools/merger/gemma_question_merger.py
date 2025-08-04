from typing import Any, Optional

from openai import OpenAI
from texttools.base.base_question_merger import BaseQuestionsMerger, MergingMode
from texttools.formatter import Gemma3Formatter

# class QuestionGeneration(BaseModel):
#     generated_question: str


class GemmaQuestionMerger(BaseQuestionsMerger):
    """
    Questions merger for Gemma-style models with one mode for now:
    1. merge the provided questions, preserving all the main points.
    Outputs JSON with a single string field: {"merged_question": "..."}.

    Allows optional extra instructions via `prompt_template`.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        chat_formatter: Optional[Any] = None,
        use_reason: bool = False,
        temperature: float = 0.5,
        prompt_template: Optional[str] = None,
        handlers: Optional[list[Any]] = None,
        **client_kwargs: Any,
    ):
        super().__init__(handlers)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs

        self.chat_formatter = chat_formatter or Gemma3Formatter()

        self.use_reason = use_reason
        self.reason_summary = None
        self.prompt_template = prompt_template

        self.json_schema = {"rewritten_question": "string"}

    def _build_messages(
        self,
        questions: list[str],
        mode: MergingMode,
    ) -> list[dict[str, str]]:
        """
        Builds the message list for the LLM API call for question merging,
        adapting the prompt based on the chosen mode.
        """
        clean_questions = self.preprocess(questions)
        messages: list[dict[str, str]] = []

        if self.prompt_template:
            messages.append({"role": "user", "content": self.prompt_template})

        if self.reason_summary:
            messages.append(
                {
                    "role": "user",
                    "content": f"Based on this analysis: {self.reason_summary}",
                }
            )

        if mode == MergingMode.DEFAULT_MODE:
            instruction = (
                "You are a language expert."
                "I will give you a list of questions that are semantically similar."
                "Your task is to merge them into one unified question that:"
                "- Preserves all the information and intent from the original questions."
                "- Sounds natural, fluent, and concise."
                "- Avoids redundancy or unnecessary repetition."
                "- Does not omit any unique idea from the originals."
                "**Output only the merged question.**"
            )
        elif mode == MergingMode.REASON_MODE:
            instruction = (
                "You are an AI assistant helping to unify semantically similar questions."
                "First, briefly extract the unique intent or content from each input question."
                "Then, write one merged question that combines all their content clearly and naturally, without redundancy."
                "Step 1: Extract key ideas."
                "Step 2: Write the final merged question."
            )
        else:
            raise ValueError(f"Unsupported rewrite mode: {mode}")

        messages.append({"role": "user", "content": instruction})
        messages.append(
            {"role": "user", "content": f"here is the questions: {clean_questions}"}
        )

        # schema_instr = f"Respond only in JSON format: {json.dumps(self.json_schema)}"
        messages.append(
            {
                "role": "user",
                "content": """
        Respond only with the new generated question, without any additional information.
        **the generated question will be in the language of the users input**
                         """,
            }
        )

        # messages.append({"role": "assistant", "content": "{"})
        # deprecated method for structured output

        # this line will restructure the messages
        # based on the formatter that we provided
        # some models will require custom settings
        restructured = self.chat_formatter.format(messages=messages)

        return restructured

    def _reason(self, questions: list[str], mode: MergingMode) -> str:
        """
        Internal reasoning step to help the model understand the core meaning
        or structure of the question depending on the mode.
        """
        if mode == MergingMode.DEFAULT_MODE:
            reason_prompt = """
                Analyze the following questions to identify their core intent, key concepts, 
                and the specific information they are seeking.
                Provide a brief, summarized understanding of the questions' meaning that
                will help in merging and rephrasing it accurately without changing its intent.
                
                **respond in the language of the question**
                """
        elif mode == MergingMode.REASON_MODE:
            reason_prompt = """
                Analyze the following questions to identify their exact wording, phrasing,
                and the literal meaning it conveys.
                Provide a brief, summarized analysis of their linguistic structure and current meaning,
                which will then be used to create a new question containing all of their contents.
                **respond in the language of the question**
                """
        else:
            raise ValueError(f"Unsupported rewrite mode for reason: {mode}")

        messages = [
            {"role": "user", "content": reason_prompt},
            {"role": "user", "content": f"here is the question: {questions}"},
        ]

        restructured = self.chat_formatter.format(messages=messages)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=restructured,
            temperature=self.temperature,
            **self.client_kwargs,
        )

        reason_summary = resp.choices[0].message.content.strip()
        self.reason_summary = reason_summary

    def rewrite_questions(
        self,
        questions: list[str],
        mode: MergingMode = MergingMode.DEFAULT_MODE,
        reason_summary: str = None,
    ) -> str:
        """
        merging the input `questions` based on the specified `mode`.
        Optionally uses an internal reasoning step for better accuracy.
        """

        if self.use_reason and not reason_summary:
            self._reason(questions, mode)
        elif reason_summary:
            self.reason_summary = reason_summary

        messages = self._build_messages(questions, mode)

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
                "original_questions": questions,
                "merged_question": result,
                "mode": mode.value,
            }
        )
        return result

    def get_reason(self):
        return self.reason_summary
