import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from texttools import TheTool

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
tool = TheTool(client=client, model="gpt-4o-mini")

# Sample inputs for stress test
inputs = [
    "Tomorrow we will be dead by the car crash",
    "Python is fun",
    "The weather is nice",
    "Thread safety test",
    "Concurrency is tricky",
    "OpenAI API is powerful",
    "Testing mutable state in classes",
    "Random philosophical text",
    "Edge case: !@#$%^&*()",
    "Another example sentence",
]

# Thread-safe storage for results and errors
results_lock = threading.Lock()
results = []
errors = []


# Worker function for threads
def run_tool(text, index):
    try:
        output = tool.extract_keywords(text, logprobs=True)["result"]
        with results_lock:
            results.append((index, text, output))
    except Exception as e:
        with results_lock:
            errors.append((index, text, str(e)))


# Number of concurrent threads
NUM_THREADS = 200

# Create a list of tasks (repeating inputs to increase stress)
tasks = [(inputs[i % len(inputs)], i) for i in range(NUM_THREADS)]

# Run tasks concurrently
with ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(run_tool, text, idx) for text, idx in tasks]
    for future in as_completed(futures):
        pass  # results and errors are already collected

# Sort results by index to maintain input order
results.sort(key=lambda x: x[0])

# Print summary
print(f"Completed {len(results)} threads successfully")
if errors:
    print(f"Encountered {len(errors)} errors:")
    for idx, text, err in errors:
        print(f"Thread {idx} with input '{text}' failed: {err}")
else:
    print("No errors detected!")

# Print sample outputs
for idx, text, output in results[:10]:
    print(f"{text} -> {output}")
