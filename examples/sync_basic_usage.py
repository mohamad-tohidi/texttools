import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from texttools import CategoryTree, TheTool

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

client = OpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)

t = TheTool(client=client, model=MODEL)


def main():
    # Categorizer: list mode
    category = t.categorize(
        "سلام حالت چطوره؟", categories=["هیچکدام", "دینی", "فلسفه"], priority=3
    )
    print(repr(category))

    # Categorizer: tree mode
    tree = CategoryTree()
    tree.add_node("اخلاق", "root")
    tree.add_node("معرفت شناسی", "root")
    tree.add_node("متافیزیک", "root", description="اراده قدرت در حیطه متافیزیک است")
    tree.add_node("فلسفه ذهن", "root")
    tree.add_node("آگاهی", "فلسفه ذهن")
    tree.add_node("ذهن و بدن", "فلسفه ذهن")
    tree.add_node("امکان و ضرورت", "متافیزیک")

    categories = t.categorize(
        "اراده قدرت مفهومی مهم در مابعد الطبیعه است که توسط نیچه مطرح شده",
        tree,
    )
    print(repr(categories))

    keywords = t.extract_keywords(
        "Tomorrow, we will be dead by the car crash", mode="count", number_of_keywords=3
    )
    print(repr(keywords))

    entities = t.extract_entities(
        "Ali will be dead by the car crash",
    )
    print(repr(entities))

    detection = t.is_question("We will be dead by the car crash")
    print(repr(detection))

    question = t.to_question(
        "We will be dead by the car crash", mode="from_text", number_of_questions=2
    )
    print(repr(question))

    merged = t.merge_questions(
        [
            "چرا ما انسان ها، موجوداتی اجتماعی هستیم؟",
            "چرا ما باید در کنار هم زندگی کنیم؟",
        ],
        mode="simple",
    )
    print(repr(merged))

    augmentation = t.augment(
        "چرا ما انسان ها، موجوداتی اجتماعی هستیم؟",
        mode="positive",
    )
    print(repr(augmentation))

    summary = t.summarize("Tomorrow, we will be dead by the car crash")
    print(repr(summary))

    translation = t.translate("سلام حالت چطوره؟", target_lang="English")
    print(repr(translation))

    propositionize = t.propositionize(
        "جنگ جهانی دوم در سال ۱۹۳۹ آغاز شد و آلمان به لهستان حمله کرد.",
    )
    print(repr(propositionize))

    check_fact = t.is_fact(
        text="امام نهم در ایران به خاک سپرده شد",
        source_text="حرم مطهر امام رضا علیه السلام در مشهد مقدس است",
    )
    print(repr(check_fact))

    class Student(BaseModel):
        result: list[dict[str, str]]

    custom_prompt = """You are a random student information generator.
                    You have to fill the a student's information randomly.
                    They should be meaningful.
                    Create one student with these info:
                    [{"name": str}, {"age": int}, {"std_id": int}]"""

    student_info = t.run_custom(custom_prompt, Student)
    print(repr(student_info))


if __name__ == "__main__":
    main()
