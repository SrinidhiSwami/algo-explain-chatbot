"""
Algorithm Explanation Chatbot - Flask Backend
Run: python app.py  →  open http://localhost:5000
"""

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, send_from_directory
from groq import Groq
import os, re

app    = Flask(__name__, static_folder=".")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── SYSTEM PROMPT ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert algorithm teacher embedded in a chat interface.
You ONLY discuss algorithms and data structures.
If the user asks about anything else, politely decline.

DETECTING MESSAGE TYPE
----------------------
You will receive two kinds of messages:

1. EXPLAIN REQUEST  — user wants a full explanation of an algorithm.
   Respond with the full structured XML format below.

2. FOLLOW-UP / DOUBT — user is asking a question or clarifying something
   about an algorithm already discussed (e.g. "why is the time complexity O(n log n)?",
   "what happens if the array is already sorted?", "can you re-explain step 3?").
   Respond with plain conversational text — NO XML tags needed.
   Keep it clear, friendly, and concise. Use ASCII diagrams if helpful.

FULL EXPLANATION FORMAT (only for explain requests):
-----------------------------------------------------
<theory>
Theory, intuition, time/space complexity, and use-cases in clear prose.
</theory>

<example_intro>
One sentence introducing the concrete example you'll walk through.
</example_intro>

<steps>
<step>
<title>Step title</title>
<visual>ASCII diagram showing state at this step</visual>
<explanation>What is happening here</explanation>
</step>
</steps>

<summary>
Brief summary and key takeaways.
</summary>

RULES:
- Always include at least 4 steps in full explanations
- ASCII visuals use [], |, ->, ^, *, +, - characters only
- No markdown inside XML tags
- For follow-ups: reply in plain text, warm and concise"""

# ── PARSERS ─────────────────────────────────────────────────────────────────
def extract_tag(text, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""

def extract_steps(text):
    return [
        {
            "title":       extract_tag(s, "title"),
            "visual":      extract_tag(s, "visual"),
            "explanation": extract_tag(s, "explanation")
        }
        for s in re.findall(r"<step>(.*?)</step>", text, re.DOTALL)
    ]

def is_structured(text):
    return "<theory>" in text and "<steps>" in text

def parse_response(text):
    if is_structured(text):
        return {
            "type":         "explanation",
            "theory":       extract_tag(text, "theory"),
            "example_intro":extract_tag(text, "example_intro"),
            "steps":        extract_steps(text),
            "summary":      extract_tag(text, "summary"),
        }
    return {"type": "followup", "text": text}

# ── ROUTES ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    body    = request.json
    message = body.get("message", "").strip()
    history = body.get("history", [])      # full conversation history from client

    if not message:
        return jsonify({"error": "Empty message"}), 400

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": message}
    ]

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
        )
        raw    = resp.choices[0].message.content
        parsed = parse_response(raw)
        parsed["raw"] = raw
        return jsonify({"success": True, "data": parsed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n🤖 AlgoBot running → http://localhost:5000\n")
    app.run(debug=True, port=5000)
