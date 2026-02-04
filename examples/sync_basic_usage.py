import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from texttools import TheTool

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

# Set logger level
logging.basicConfig(level=logging.INFO)

# Initialize clients
client = OpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)
the_tool = TheTool(client=client, model=MODEL, raise_on_error=False)


def main():
    category = the_tool.categorize(
        "انسان‌ها به چه دلایلی هنر را خلق می‌کنند و چه تاثیری بر جامعه دارد؟",
        categories=["هیچکدام", "فلسفه", "علوم تجربی", "علوم اجتماعی"],
    )
    print(category.to_json())

    keywords = the_tool.extract_keywords(
        "Climate change is causing unprecedented weather patterns, rising sea levels, and global ecological disruptions.",
        mode="count",
        number_of_keywords=3,
    )
    print(keywords.to_json())

    entities = the_tool.extract_entities(
        "Albert Einstein developed the theory of relativity, which revolutionized modern physics.",
    )
    print(entities.to_json())

    is_question = the_tool.is_question("What are the main causes of global warming?")
    print(is_question.to_json())

    questions = the_tool.to_question(
        "Renewable energy sources like solar and wind are essential to reduce carbon emissions.",
        mode="from_text",
        number_of_questions=2,
    )
    print(questions.to_json())

    merged_question = the_tool.merge_questions(
        [
            "چرا انسان‌ها در طول تاریخ تمدن‌ها تمدن ساخته‌اند؟",
            "چه عواملی باعث پیشرفت فرهنگی و علمی جوامع شده‌اند؟",
        ],
        mode="simple",
    )
    print(merged_question.to_json())

    augmented = the_tool.augment(
        "انسان‌ها به دلیل نیاز به شناخت جهان و تعامل اجتماعی، هنر و زبان را توسعه داده‌اند.",
        mode="positive",
    )
    print(augmented.to_json())

    summary = the_tool.summarize(
        "The novel explores the psychological effects of isolation during long space missions and the human drive to adapt and survive."
    )
    print(summary.to_json())

    translation = the_tool.translate(
        "علم و تکنولوژی می‌تواند زندگی انسان‌ها را به شکل چشمگیری بهبود دهد.",
        target_lang="English",
    )
    print(translation.to_json())

    propositions = the_tool.propositionize(
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics."
    )
    print(propositions.to_json())

    is_fact = the_tool.is_fact(
        text="The Great Wall of China stretches over 21,000 kilometers.",
        source_text="Historical surveys indicate that the Great Wall of China extends approximately 21,196 kilometers across northern China.",
    )
    print(is_fact.to_json())


if __name__ == "__main__":
    main()
