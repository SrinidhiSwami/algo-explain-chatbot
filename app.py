"""
Algorithm Explanation Chatbot - Flask Backend
Run: python app.py
Then open: http://localhost:5000
"""

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, send_from_directory
from groq import Groq
import os
import re

app = Flask(__name__, static_folder=".")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an expert algorithm teacher. You ONLY explain algorithms.
If the user asks about anything other than algorithms or data structures, politely decline
and ask them to ask about an algorithm.

When explaining an algorithm, ALWAYS follow this EXACT structured format using XML tags:

<theory>
Explain the theory, intuition, time/space complexity, and use-cases in clear prose.
</theory>

<example_intro>
Briefly introduce the concrete example you will walk through.
</example_intro>

<steps>
<step>
<title>Step title here</title>
<visual>ASCII art or text visualization showing the state at this step</visual>
<explanation>Clear explanation of what is happening in this step</explanation>
</step>
</steps>

<summary>
A brief summary of what we learned and key takeaways.
</summary>

Rules:
- Always include at least 4 steps
- Keep visuals as clean ASCII diagrams using characters like [], |, ->, ^, *, etc.
- Each step must be self-contained and show clear progress
- Do NOT use markdown inside the XML tags"""


def extract_tag(text, tag):
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_steps(text):
    steps_raw = re.findall(r"<step>(.*?)</step>", text, re.DOTALL)
    steps = []
    for s in steps_raw:
        steps.append({
            "title": extract_tag(s, "title"),
            "visual": extract_tag(s, "visual"),
            "explanation": extract_tag(s, "explanation")
        })
    return steps


def parse_response(text):
    return {
        "theory": extract_tag(text, "theory"),
        "example_intro": extract_tag(text, "example_intro"),
        "steps": extract_steps(text),
        "summary": extract_tag(text, "summary"),
        "raw": text
    }


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/explain", methods=["POST"])
def explain():
    data = request.json
    algorithm = data.get("algorithm", "").strip()
    history = data.get("history", [])

    if not algorithm:
        return jsonify({"error": "No algorithm provided"}), 400

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": f"Please explain the following algorithm: {algorithm}"}
    ]

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
        )
        raw = response.choices[0].message.content
        parsed = parse_response(raw)
        return jsonify({"success": True, "data": parsed, "raw": raw})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
