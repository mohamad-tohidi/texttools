from openai import OpenAI

from texttools.tools.question_generator.question_generator import QuestionGenerator

client = OpenAI(
    base_url="http://185.208.182.157:9310/v1", api_key="hamta_T0k3n879875412"
)

tool = QuestionGenerator(client=client, model="gemma-3", use_reason=False)
c = tool.generate_question("The capital of France is Paris.")
print(c)
