import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from texttools import TheTool

# Load environment variables from .env
load_dotenv()

# Load API key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")
model = "google/gemma-3n-e4b-it"

# Create OpenAI client
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

# Create an instance of TheTool
t = TheTool(client=client)

# Categorizer
category = t.categorize("سلام حالت چطوره؟")
print(category)

# Keyword Extractor
keywords = t.extract_keywords("Tomorrow, we will be dead by the car crash")
print(keywords)

# NER Extractor
entities = t.extract_entities("We will be dead by the car crash")
print(entities)

# Question Detector
detection = t.detect_question("We will be dead by the car crash")
print(detection)

# Question from Answer Generator
question = t.generate_question_from_text("We will be dead by the car crash")
print(question)

# Question Merger
question = t.merge_questions(
    ["چرا ما انسان ها، موجوداتی اجتماعی هستیم؟", "چرا ما باید در کنار هم زندگی کنیم؟"],
    mode="default",
)
print(question)

# Rewriter
question = t.rewrite(
    "چرا ما انسان ها، موجوداتی اجتماعی هستیم؟",
    mode="same_meaning_different_wording",
)
print(question)

# Question Generator from Subject
questions = t.generate_questions_from_subject("Friendship", 3)
print(questions)

# Summarizer
summary = t.summarize("Tomorrow, we will be dead by the car crash")
print(summary)

# Translator
translation = t.translate("سلام حالت چطوره؟", target_language="English")
print(translation)


# Custom tool
class Student(BaseModel):
    result: list[dict[str, str]]


custom_prompt = """You are a random student information generator.
                   You have to fill the a student's information randomly.
                   They should be meaningful.
                   Create one student with these info:
                   [{"name": str}, {"age": int}, {"std_id": int}]"""

student_info = t.custom_tool(custom_prompt, Student)
print(student_info)
