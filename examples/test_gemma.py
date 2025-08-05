import os

os.environ["OPENAI_API_KEY"] = "AIzaSyBUvUkMOMJHmHPQDGI_fKVZnK-4yk1QR6s"
os.environ["OPENAI_BASE_URL"] = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)

from openai import OpenAI
import json
import httpx
from dotenv import load_dotenv


load_dotenv()

proxy_url = os.getenv("PROXY")

httpx_client = httpx.Client(proxy=proxy_url)


try:
    r = httpx_client.get("https://openrouter.ai/api/v1/models")
    print(r.status_code)
    print(r.text)
except httpx.ProxyError as e:
    print("ProxyError:", e)


client = OpenAI(
    http_client=httpx_client
    # api_key=API_KEY,
    # base_url="https://openrouter.ai/api/v1"
)


json_schema = {"is_question": bool}


text = "the text that i want to detect if it is question or not"


messages = [
    {
        "role": "user",
        "content": f"respond only in JSON format, in this form {json_schema}",
    },
    {"role": "user", "content": ""},
    {"role": "assistant", "content": "{"},
]

response = client.chat.completions.create(model="gemma-3n-e4b-it", messages=messages)

result = response.choices[0].message.content
result = "{" + result

json.loads(result)
