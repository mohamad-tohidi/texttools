import os
import sys

# === ۱. رفع مشکل ImportError در محیط توسعه ===
# این خط مسیر دایرکتوری ریشه پروژه را (که حاوی پوشه texttools است)
# به ابتدای مسیر جستجوی پایتون اضافه می کند.
# این تضمین می کند که پایتون فایل های ویرایش شده محلی شما را لود کند.
# فرض بر این است که test_all_tools.py در پوشه texttools/tests/ قرار دارد (یا مشابه آن).
# اگر test_all_tools.py در کنار پوشه texttools است، از sys.path.insert(0, os.path.abspath(os.path.dirname(__file__))) استفاده کنید.
# اما با توجه به نام فایل، احتمالا این ساختار درست است:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# -------------------------------------------------------------------


from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from texttools import TheTool # این حالا فایل محلی شما را می شناسد

# Load environment variables from .env
load_dotenv()

# اگر از gemma3 استفاده می‌کنید و این در .env نیست، اینجا آن را تنظیم کنید:
# MODEL = os.getenv("MODEL", "gemma3")
MODEL = os.getenv("MODEL")


# === ۲. تعریف متن تست جدید ===
TEST_TEXT_FA = (
''' 
پیامبر (ص) فرمود: بودن گوسفند در خانه باعث می شود هفتاد در فقر از بین برود.
'''
)
# -----------------------------


# Create OpenAI client
client = OpenAI

# Create an instance of TheTool
t = TheTool(client=client, model=MODEL)

# Categorizer
category = t.categorize("سلام حالت چطوره؟")
print(repr(category))

# Keyword Extractor
keywords = t.extract_keywords("Tomorrow, we will be dead by the car crash")
print(repr(keywords))

# NER Extractor (قبلاً بود)
entities = t.extract_entities("We will be dead by the car crash")
print(repr(entities))

# === ۳. فراخوانی Entity Detector (ابزار جدید شما) ===
entity_detector_result = t.entity_detector(text=TEST_TEXT_FA)
print(repr(entity_detector_result))
# ----------------------------------------------------


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
    with_analysis=True,
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