from openai import OpenAI

from texttools.tools.question_rewriter.question_rewriter import QuestionRewriter

client = OpenAI(
    base_url="http://185.208.182.157:9310/v1", api_key="hamta_T0k3n879875412"
)

tool = QuestionRewriter(
    client=client,
    model="gemma-3",
    use_reason=True,
    mode="same_meaning_different_wording_mode",
)
c = tool.rewrite_question(
    "What is the benefit of having a lot of friends?",
)
print(c)
