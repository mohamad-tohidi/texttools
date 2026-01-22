import os

from dotenv import load_dotenv
from openai import OpenAI
from texttools import TheTool

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

client = OpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)

the_tool = TheTool(client=client, model=MODEL)


def main():
    category = the_tool.categorize(
        "کانت در بسیاری از موارد اشتباه میکرد",
        categories=["هیچکدام", "دینی", "فلسفه"],
        with_analysis=True,
        user_prompt="Consider proper names carefully",
        logprobs=True,
        top_logprobs=3,
        priority=3,
    )
    print(category.to_json())

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


    class Student(BaseModel):
        result: list[dict[str, str]]

    custom_prompt = """You are a random student information generator.
                    You have to fill the a student's information randomly.
                    They should be meaningful.
                    Create one student with these info:
                    [{"name": str}, {"age": int}, {"std_id": int}]"""
    
    student = the_tool.run_custom(custom_prompt, Student)
    print(student.to_json())


if __name__ == "__main__":
    main()
