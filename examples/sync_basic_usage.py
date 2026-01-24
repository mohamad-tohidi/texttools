import os

from dotenv import load_dotenv
from openai import OpenAI
from texttools import TheTool

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

client = OpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)

the_tool = TheTool(client=client, model=MODEL)


def main():
    outputs = []

    outputs.append(the_tool.categorize(
        "سلام حالت چطوره؟",
        categories=["هیچکدام", "دینی", "فلسفه"],
    ))

    outputs.append(the_tool.extract_keywords(
        "Tomorrow, we will be dead by the car crash", mode="count", number_of_keywords=3
    ))

    outputs.append(the_tool.extract_entities(
        "Ali will be dead by the car crash",
    ))

    outputs.append(the_tool.is_question("We will be dead by the car crash"))

    outputs.append(the_tool.to_question(
        "We will be dead by the car crash", mode="from_text", number_of_questions=2
    ))

    outputs.append(the_tool.merge_questions(
        [
            "چرا ما انسان ها، موجوداتی اجتماعی هستیم؟",
            "چرا ما باید در کنار هم زندگی کنیم؟",
        ],
        mode="simple",
    ))

    outputs.append(the_tool.augment(
        "چرا ما انسان ها، موجوداتی اجتماعی هستیم؟",
        mode="positive",
    ))

    outputs.append(the_tool.summarize("Tomorrow, we will be dead by the car crash"))

    outputs.append(the_tool.translate("سلام حالت چطوره؟", target_lang="English"))

    outputs.append(the_tool.propositionize(
        "جنگ جهانی دوم در سال ۱۹۳۹ آغاز شد و آلمان به لهستان حمله کرد.",
    ))

    outputs.append(the_tool.is_fact(
        text="امام نهم در ایران به خاک سپرده شد",
        source_text="حرم مطهر امام رضا علیه السلام در مشهد مقدس است",
    ))

    for output in outputs:
        print(output.to_json())


if __name__ == "__main__":
    main()
