import os

from dotenv import load_dotenv
from openai import OpenAI

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

# Create a category tree
tree = CategoryTree("category_test_tree")
tree.add_category("اخلاق")
tree.add_category("معرفتشناسی")
tree.add_category("متافیزیک")
tree.add_category("فلسفه ذهن")
tree.add_category("آگاهی", "فلسفه ذهن")
tree.add_category("ذهن و بدن", "فلسفه ذهن")
tree.add_category("امکان و ضرورت", "متافیزیک")

categories = t.categorize(
    "اراده قدرت مفهومی مهم در مابعد الطبیعه است که توسط نیچه مطرح شده",
    tree,
)
print(repr(categories))
