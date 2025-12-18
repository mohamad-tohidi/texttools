import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from texttools import TheTool, CategoryTree

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

client = OpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)

t = TheTool(client=client, model=MODEL)

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
tree.add_node(
    "فلسفه ذهن", "root", description="فلسفه ذهن به چگونگی درک ما از جهان می پردازد"
)
tree.add_node("آگاهی", "فلسفه ذهن")
tree.add_node("ذهن و بدن", "فلسفه ذهن")
tree.add_node("امکان و ضرورت", "متافیزیک")

categories = t.categorize(
    "اراده قدرت مفهومی مهم در مابعد الطبیعه است که توسط نیچه مطرح شده",
    tree,
)
print(repr(categories))

# Keyword Extractor
keywords = t.extract_keywords(
    "Tomorrow, we will be dead by the car crash", mode="count", number_of_keywords=3
)
print(repr(keywords))

# NER Extractor
entities = t.extract_entities(
    "Ali will be dead by the car crash",
    entities=["EVENT"],
    with_analysis=True,
)
print(repr(entities))


# Question Detector
detection = t.is_question("We will be dead by the car crash")
print(repr(detection))

# Question from Text Generator
question = t.text_to_question("We will be dead by the car crash", 2)
print(repr(question))

# Question Merger
merged = t.merge_questions(
    ["چرا ما انسان ها، موجوداتی اجتماعی هستیم؟", "چرا ما باید در کنار هم زندگی کنیم؟"],
    mode="default",
)
print(repr(merged))

# Rewriter
rewritten = t.rewrite(
    "چرا ما انسان ها، موجوداتی اجتماعی هستیم؟",
    mode="positive",
)
print(repr(rewritten))

# Question from Subject Generator
questions = t.subject_to_question("Friendship", 3)
print(repr(questions))

# Summarizer
summary = t.summarize("Tomorrow, we will be dead by the car crash")
print(repr(summary))

# Translator
translation = t.translate("سلام حالت چطوره؟", target_language="English")
print(repr(translation))

# Propositionizer
propositionize = t.propositionize(
    "جنگ جهانی دوم در سال ۱۹۳۹ آغاز شد و آلمان به لهستان حمله کرد.",
    output_lang="Persian",
)
print(repr(propositionize))

# Check Fact
check_fact = t.check_fact(
    text="امام نهم در ایران به خاک سپرده شد",
    source_text="حرم مطهر امام رضا علیه السلام در مشهد مقدس هست",
)
print(repr(check_fact))


# Run Custom
class Student(BaseModel):
    result: list[dict[str, str]]


custom_prompt = """You are a random student information generator.
                   You have to fill the a student's information randomly.
                   They should be meaningful.
                   Create one student with these info:
                   [{"name": str}, {"age": int}, {"std_id": int}]"""

student_info = t.run_custom(custom_prompt, Student)
print(repr(student_info))
