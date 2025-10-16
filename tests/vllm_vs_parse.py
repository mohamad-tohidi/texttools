import os

from dotenv import load_dotenv
from openai import OpenAI

from texttools import TheTool

# Load environment variables from .env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

model = "google/gemma-3n-e4b-it"

# Create OpenAI client
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Create a parser instance of TheTool
parse_tool = TheTool(client=client, model=model)
parse_tool.operator.RESP_FORMAT = "parse"

# Create a vllm structured output instance of TheTool
vllm_tool = TheTool(client=client, model=model)
vllm_tool.operator.RESP_FORMAT = "vllm"

# Define test inputs
cat_text = "سلام حالت چطوره؟"
key_text = "Tomorrow, we will be dead by the car crash"
ner_text = "We will be dead by the car crash"
qd_text = "We will be dead by the car crash"
qfa_text = "We will be dead by the car crash"
merge_text = [
    "چرا ما انسان ها، موجوداتی اجتماعی هستیم؟",
    "چرا ما باید در کنار هم زندگی کنیم؟",
]
rewrite_text = "چرا ما انسان ها، موجوداتی اجتماعی هستیم؟"
qfs_text = "Friendship"
summary_text = "با توجه به تعداد روزهای متعددی که جلسات را با افتخار برگزار میکنیم، باید توجه بنمایید که نباید درگیر مسائل جزئی که میتواند ما را از حل مسئله دور کند توجه کنیم."
translate_text = "سلام حالت چطوره؟"

# Categorizer
parse = parse_tool.categorize(cat_text)
vllm = vllm_tool.categorize(cat_text)
print(f"Categorizer\nParse: {parse}\nvllm:  {vllm}\n\n")

# Keyword Extractor
parse = parse_tool.extract_keywords(key_text)
vllm = vllm_tool.extract_keywords(key_text)
print(f"Keyword Extractor\nParse: {parse}\nvllm:  {vllm}\n\n")

# NER Extractor
parse = parse_tool.extract_entities(ner_text)
vllm = vllm_tool.extract_entities(ner_text)
print(f"NER Extractor\nParse: {parse}\nvllm:  {vllm}\n\n")

# Question Detector
parse = parse_tool.is_question(qd_text)
vllm = vllm_tool.is_question(qd_text)
print(f"Question Detector\nParse: {parse}\nvllm:  {vllm}\n\n")

# Question from Answer Generator
parse = parse_tool.text_to_question(qfa_text)
vllm = vllm_tool.text_to_question(qfa_text)
print(f"Question from Answer Generator\nParse: {parse}\nvllm:  {vllm}\n\n")

# Question Merger
parse = parse_tool.merge_questions(merge_text, mode="default")
vllm = vllm_tool.merge_questions(merge_text, mode="default")
print(f"Question Merger\nParse: {parse}\nvllm:  {vllm}\n\n")

# Question Rewriter
parse = parse_tool.rewrite(rewrite_text, mode="positive")
vllm = vllm_tool.rewrite(rewrite_text, mode="positive")
print(f"Question Rewriter\nParse: {parse}\nvllm:  {vllm}\n\n")

# Question Generator from Subject
parse = parse_tool.subject_to_question(qfs_text, 3)
vllm = vllm_tool.subject_to_question(qfs_text, 3)
print(f"Question Generator from Subject\nParse: {parse}\nvllm:  {vllm}\n\n")

# Summarizer
parse = parse_tool.summarize(summary_text)
vllm = vllm_tool.summarize(summary_text)
print(f"Summarizer\nParse: {parse}\nvllm:  {vllm}\n\n")

# Translator
parse = parse_tool.translate(translate_text, target_language="English")
vllm = vllm_tool.translate(translate_text, target_language="English")
print(f"Translator\nParse: {parse}\nvllm:  {vllm}\n\n")
