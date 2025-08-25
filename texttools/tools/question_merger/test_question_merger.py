from openai import OpenAI

from texttools.tools.question_merger.question_merger import QuestionMerger

client = OpenAI(
    base_url="http://185.208.182.157:9310/v1", api_key="hamta_T0k3n879875412"
)

tool = QuestionMerger(
    client=client, model="gemma-3", use_reason=False, mode="default_mode"
)
c = tool.merge(
    [
        "Is having a lot of friends good?",
        "Should I choose a lot amount of freinds?",
        "What is the benefit of having a lot of friends?",
    ]
)
print(c)
