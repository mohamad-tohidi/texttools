from openai import OpenAI

from texttools.tools.question_detector.question_detector import QuestionDetector

client = OpenAI(
    base_url="http://185.208.182.157:9310/v1", api_key="hamta_T0k3n879875412"
)

tool = QuestionDetector(client=client, model="gemma-3", use_reason=False)
c = tool.detect("پایتخت فرانسه چیست؟")
print(c)
