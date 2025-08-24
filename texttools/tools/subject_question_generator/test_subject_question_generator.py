from openai import OpenAI

from texttools.tools.subject_question_generator.subject_question_generator import (
    SubjectQuestionGenerator,
)


client = OpenAI(
    base_url="http://185.208.182.157:9310/v1", api_key="hamta_T0k3n879875412"
)

tool = SubjectQuestionGenerator(client=client, model="gemma-3", use_reason=False)
c = tool.generate_question("داشتن دوستان زیاد", "English", "5")
print(c)
