from typing import Any, Optional

from openai import OpenAI

from texttools.base.base_question_generator import BaseQuestionGenerator
from texttools.formatter import Gemma3Formatter

# class QuestionGeneration(BaseModel):
#     generated_question: str


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
        chat_formatter: Optional[Any] = None,
        use_reason: bool = False,
        temperature: float = 0.0,
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
        self.prompt_template = prompt_template

        # Define the JSON schema for the generated question output
        self.json_schema = {"generated_question": "string"}

    def _build_messages(
        self, answer: str, reason: Optional[str] = None
    ) -> list[dict[str, str]]:
        """
        Builds the message list for the LLM API call for question generation.
        """
        clean_answer = self.preprocess(answer)
        messages: list[dict[str, str]] = []

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
                "content": """Given the following answer, generate a single, 
                appropriate question that this answer would directly respond to.
                the generated answer should be independently meaningful,
                and not mentioning any verbs like, this, that, he or she ... on the question.
                # **the generated question will be in the language of the users input**
                
                """,
            }
        )
        messages.append(
            {"role": "user", "content": f"here is the text: {clean_answer}"}
        )

        # Ensure the schema is dumped as a valid JSON string for the LLM
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

        # messages.append(
        #     {"role": "assistant", "content": "{\n"}
        # )  # Hint to start JSON output
        # in this new version we will use
        # parse function of openai library

        # this line will restructure the messages
        # based on the formatter that we provided
        # some models will require custom settings
        restructured = self.chat_formatter.format(messages=messages)

        return restructured

    def _reason(self, answer: str) -> str:
        """
        Internal reasoning step to help the model understand the core information
        and implications of the answer.
        """
        messages = [
            {
                "role": "user",
                "content": """
                    Analyze the following answer to identify its key facts,
                    main subject, and what kind of information it provides.
                    Provide a brief, summarized understanding of the answer's content that will 
                    help in formulating a relevant and direct question.
                    
                    provide the summary in the language of the content.
                    just mention the keypoints that was provided in the answer
                    
                    
                    """,
            },
            {
                "role": "user",
                "content": f"""
                
                    Here is the content:
                    
                    {answer}
                    
                    respond only with the language of the content
                    """,
            },
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

    def generate_question(self, answer: str) -> str:
        """
        Generates a question for the input `answer`.
        Optionally uses an internal reasoning step for better accuracy.
        """
        reason_summary = None
        if self.use_reason:
            reason_summary = self._reason(answer)

        messages = self._build_messages(answer, reason_summary)

        # i am deprecating the usage of structured output in the tasks that
        # the input and output is str
        # as we have noticed a huge decrease in the models outputs quality

        #
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
                "original_answer": answer,
                "generated_question": result,
            }
        )
        return result
