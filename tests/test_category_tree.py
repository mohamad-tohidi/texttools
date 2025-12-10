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
tree.add_node("اخلاق")
tree.add_node("معرفت شناسی")
tree.add_node("متافیزیک", description="اراده قدرت در حیطه متافیزیک است")
tree.add_node("فلسفه ذهن", description="فلسفه ذهن به چگونگی درک ما از جهان می پردازد")
tree.add_node("آگاهی", "فلسفه ذهن", description="آگاهی خیلی مهم است")
tree.add_node("ذهن و بدن", "فلسفه ذهن")
tree.add_node("امکان و ضرورت", "متافیزیک")

# Test category tree
categories = t.categorize(
    "اراده قدرت مفهومی مهم در مابعد الطبیعه است که توسط نیچه مطرح شده",
    tree,
    mode="category_tree",
)
print(repr(categories))

# Create category list
category_list = ["اخلاق", "معرفت شناسی", "متافیزیک", "فلسفه ذهن"]

# Test list mode
category = t.categorize(
    "اراده قدرت مفهومی مهم در مابعد الطبیعه است که توسط نیچه مطرح شده",
    category_list,
    mode="category_list",
)
print(repr(category))
