import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from x import x

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
    translation = the_tool.translate(
        x,
        target_language="English",
    )
    print(translation.to_json())


if __name__ == "__main__":
    main()
    print(len(x.split(" ")))
