from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel

from texttools.base.base_question_generator import BaseQuestionGeneratorFromSubject
from texttools.formatter import Gemma3Formatter


class QuestionGeneration(BaseModel):
    """
    we use this structue, the model will feel this class
    """

    reasoning_summary: str
    questions: list


class GemmaQuestionGeneratorFromSubject(BaseQuestionGeneratorFromSubject):
    """
    Question Generator for Gemma-style models with optional reasoning step.

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

        # [DEPRECATED] we wont use unncessery structured outputs anymore
        # Define the JSON schema for the generated question output
        # self.json_schema = {"generated_question": "string"}

    def _build_messages(
        self,
        subject: str,
        reason: Optional[str] = None,
        number_of_questions: int = 5,
        language: str = "farsi/Persian",
    ) -> List[Dict[str, str]]:
        """
        Builds the message list for the LLM API call for question generation.
        """
        clean_subject = self.preprocess(subject)
        messages: List[Dict[str, str]] = []

        if self.prompt_template:
            messages.append({"role": "user", "content": self.prompt_template})

        if reason:
            messages.append(
                {
                    "role": "user",
                    "content": f"Based on this analysis of the subject: {reason}",
                }
            )

        messages.append(
            {
                "role": "user",
                "content": f"""Given the following subject, generate a single,
                appropriate question that this subject would directly respond to.
                the generated subject should be independently meaningful,
                and not mentioning any verbs like, this, that, he or she ... on the question.
                **the generated question will be in this language {language}**

                """,
            }
        )
        messages.append(
            {"role": "user", "content": f"here is the text: {clean_subject}"}
        )

        # Ensure the schema is dumped as a valid JSON string for the LLM
        # schema_instr = f"Respond only in JSON format: {json.dumps(self.json_schema)}"
        messages.append(
            {
                "role": "user",
                "content": f"""
        Respond only with the new generated question, without any additional information.
        **the generated question will be in this language {language}**
        generate {number_of_questions} number os question in the questions list.
        
        You must return ONLY a single JSON object that matches the schema.
        Do NOT include any explanation before or after the JSON.
        End the JSON with a closing brace }} and nothing else.
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

    def _reason(self, subject: str, language: str) -> str:
        """
        Internal reasoning step to help the model understand the core information
        and implications of the subject.
        """
        messages = [
            {
                "role": "user",
                "content": """
                    our goal is to generate questions, from the given subject that the user has provided
                    the questions must be meaningfull, some of them should be specific and some should be general
                    but first, in this step we want to analyze the inputted subject that the user asked us to generate questions
                    for it

                    what is the subject
                    we need summerized analysis of the input subject
                    what point of views can we see it and generate questoins from it

                    questions that real users might have


                    """,
            },
            {
                "role": "user",
                "content": f"""

                    Here is the subject:

                    {subject}

                    respond only with this language {language}

                    """,
            },
            # {
            #     "role": "assistant",
            #     "content": """
            #         Sure, here is a summerized analysis
            #     """,
            # },
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

    def generate_question(
        self, subject: str, number_of_questions: int, language: str
    ) -> str:
        """
        Generates a question for the input `subject`.
        Optionally uses an internal reasoning step for better accuracy.

        language: the language of the question

        """
        reason_summary = None
        if self.use_reason:
            reason_summary = self._reason(subject, language)

        messages = self._build_messages(
            subject, reason_summary, number_of_questions, language
        )

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=QuestionGeneration,
            temperature=self.temperature,
            extra_body=dict(
                guided_decoding_backend="auto",
            ),
            **self.client_kwargs,
        )
        message = completion.choices[0].message
        if message.parsed:
            result = message.parsed.questions
        else:
            raise ValueError(
                f"Failed to parse the response. Raw content: {message.content}"
            )

        # dispatch and return
        self._dispatch(
            {
                "original_subject": subject,
                "generated_question": result,
            }
        )
        return result
