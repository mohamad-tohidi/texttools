from texttools.tools.question_detector import GemmaQuestionDetector
import json
 
# ────────────────────────────────────────────────────────────────────────────────
# (3) Test function that calls _build_messages and prints the result or exception
# ────────────────────────────────────────────────────────────────────────────────

def test_build_messages():
    print("\n==== Test: No reason, no prompt_template ====")
    detector1 = GemmaQuestionDetector(model="gemma-3")
    try:
        msgs1 = detector1._build_messages("Hello, is this working?", reason=None)
        print(json.dumps(msgs1, indent=2))
    except Exception as e:
        print("Error:", e)

    print("\n==== Test: With prompt_template, no reason ====")
    detector2 = GemmaQuestionDetector(model="gemma-3", prompt_template="Please check if this is a question.")
    try:
        msgs2 = detector2._build_messages("What's your name?", reason=None)
        print(json.dumps(msgs2, indent=2))
    except Exception as e:
        print("Error:", e)

    print("\n==== Test: With reason and prompt_template ====")
    detector3 = GemmaQuestionDetector(model="gemma-3", prompt_template="Check it carefully.")
    try:
        msgs3 = detector3._build_messages("Is Python your favorite language?", reason="I need to be sure this is a question.")
        print(json.dumps(msgs3, indent=2))
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    test_build_messages()
