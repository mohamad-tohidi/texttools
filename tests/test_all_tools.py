import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from texttools import TheTool, CategoryTree

# Load environment variables from .env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

# Create OpenAI client
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Create an instance of TheTool
t = TheTool(client=client, model=MODEL)

# Categorizer: list mode
category = t.categorize("سلام حالت چطوره؟", categories=["هیچکدام", "دینی", "فلسفه"])
print(repr(category))

# Categorizer: tree mode
tree = CategoryTree("category_test_tree")
tree.add_node("اخلاق")
tree.add_node("معرفت شناسی")
tree.add_node("متافیزیک", description="اراده قدرت در حیطه متافیزیک است")
tree.add_node("فلسفه ذهن", description="فلسفه ذهن به چگونگی درک ما از جهان می پردازد")
tree.add_node("آگاهی", "فلسفه ذهن", description="آگاهی خیلی مهم است")
tree.add_node("ذهن و بدن", "فلسفه ذهن")
tree.add_node("امکان و ضرورت", "متافیزیک")

categories = t.categorize(
    "اراده قدرت مفهومی مهم در مابعد الطبیعه است که توسط نیچه مطرح شده",
    tree,
    mode="category_tree",
)
print(repr(categories))

# Keyword Extractor
keywords = t.extract_keywords(
    "Tomorrow, we will be dead by the car crash", mode="count", number_of_keywords=3
)
print(repr(keywords))

# NER Extractor
entities = t.extract_entities("We will be dead by the car crash")
print(repr(entities))


# Question Detector
detection = t.is_question("We will be dead by the car crash")
print(repr(detection))

# Question from Text Generator
question = t.text_to_question("We will be dead by the car crash")
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

# Question Generator from Subject
questions = t.subject_to_question("Friendship", 3)
print(repr(questions))

# Summarizer
summary = t.summarize("Tomorrow, we will be dead by the car crash")
print(repr(summary))

# Translator
translation = t.translate("سلام حالت چطوره؟", target_language="English")
print(repr(translation))

# propositionize
propositionize = t.propositionize("جنگ جهانی دوم در سال ۱۹۳۹ آغاز شد و آلمان به لهستان حمله کرد.", output_lang="Persian")
print(repr(propositionize))

# fact_check
fact_check = t.fact_check(text="امام نهم در ایران به خاک سپرده شد", source_text="حرم مطهر امام رضا علیه السلام در مشهد مقدس هست")
print(repr(fact_check))

# Custom tool
class Student(BaseModel):
    result: list[dict[str, str]]


custom_prompt = """You are a random student information generator.
                   You have to fill the a student's information randomly.
                   They should be meaningful.
                   Create one student with these info:
                   [{"name": str}, {"age": int}, {"std_id": int}]"""

student_info = t.run_custom(custom_prompt, Student)
print(repr(student_info))
