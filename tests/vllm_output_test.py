import json
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI()


class Output(BaseModel):
    reason: str
    tag: Literal["Positive", "Negative"]


messages = [
    {
        "role": "user",
        "content": """
        هدف ما طبقه بندی متن هست
        متن رو بخون و ایده اصلی و آنالیزی کوتاه از اون رو ارائه بده

        بسیار خلاصه باشه خروجی تو
        نهایتا 20 کلمه

        در نهایت یکی از تگ هارو انتخاب کن
 
        متن:
        
        امروز میخواهم به خونه برگردم!!
        """,
    }
]


def run_parse():
    return client.beta.chat.completions.parse(
        model="gemma-3",
        messages=messages,
        response_format=Output,
        extra_body=dict(guided_decoding_backend="auto"),
    )


def run_json_schema():
    return client.chat.completions.create(
        model="gemma-3",
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "output-schema",
                "schema": Output.model_json_schema(),
            },
        },
        extra_body=dict(guided_decoding_backend="auto"),
    )


def main():
    # Run parse() and print JSON dict
    parsed_response = run_parse()

    parsed_response = parsed_response.choices[0].message
    parsed_response = parsed_response.parsed

    print(parsed_response)

    # Run json_schema and parse + print JSON dict
    json_schema_response = run_json_schema()
    raw_content = json_schema_response.choices[0].message.content
    content_json = json.loads(raw_content)
    print(json.dumps(content_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
