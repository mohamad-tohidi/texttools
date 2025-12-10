import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from texttools import TheTool

# Load environment variables from .env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

# Create OpenAI client
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Create an instance of TheTool
t = TheTool(client=client, model=MODEL)

# Categorizer
category = t.categorize("سلام حالت چطوره؟")
print(repr(category))

# Keyword Extractor
keywords = t.extract_keywords("Tomorrow, we will be dead by the car crash")
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
