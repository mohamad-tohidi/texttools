import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from texttools import CategoryTree, TheTool

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

# Set logger level
logging.basicConfig(level=logging.DEBUG)

# Initialize clients
client = OpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)
the_tool = TheTool(client=client, model=MODEL)


def main():
    # Run categorizer with CategoryTree
    tree = CategoryTree()
    tree.add_node("اخلاق", "root")
    tree.add_node("معرفت شناسی", "root")
    tree.add_node("متافیزیک", "root", description="اراده قدرت در حیطه متافیزیک است")
    tree.add_node("فلسفه ذهن", "root")
    tree.add_node("آگاهی", "فلسفه ذهن")
    tree.add_node("ذهن و بدن", "فلسفه ذهن")
    tree.add_node("امکان و ضرورت", "متافیزیک")

    category = the_tool.categorize(
        "اراده قدرت مفهومی مهم در مابعد الطبیعه است که توسط نیچه مطرح شده",
        tree,
        with_analysis=True,
        user_prompt="Consider proper names carefully",
        priority=3,
    )
    print(category.to_json())

    # Run keyword extractor with custom validator
    def validate(result: str) -> bool:
        return True if "هنرمندان" in result else False

    keywords = the_tool.extract_keywords(
        "تصورات اشتباه ما در باب زندگینامه هنرمندان بسیار زیاد است به طوری که اکثر مواقع در فهمیدن زندگی، آثار و دستاوردهای آنها مرتکب خطا می شویم",
        mode="count",
        number_of_keywords=4,
        output_lang="English",
        temperature=1.2,
        validator=validate,
        max_validation_retries=5,
    )
    print(keywords.to_json())

    # Run custom tool with an arbitraty structured output
    class Student(BaseModel):
        result: list[dict[str, str]]

    custom_prompt = """You are a random student information generator.
                    You have to fill the a student's information randomly.
                    They should be meaningful.
                    Each field should be filled with specified type in the prompt.
                    For example: name = Ali, age = 3, std_id = 1234
                    Create one student with these info:
                    [{"name": str}, {"age": int}, {"std_id": int}]"""

    student = the_tool.run_custom(custom_prompt, Student)
    print(student.to_json())


if __name__ == "__main__":
    main()
