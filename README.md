# AlgoBot — DSA Tutor Chatbot

A Flask-based chatbot that explains algorithms and data structures through structured, step-by-step walkthroughs — then lets you ask natural follow-up questions in the same conversation. Powered by Groq's LLaMA 3 for fast inference.

## Why this exists

Most AI chat tools give you a wall of text when you ask "explain merge sort." This project instead asks the LLM to return a **structured response** (via custom XML-style tags) that the frontend parses and renders as an animated, step-by-step card — theory, then a worked example, then each step with an ASCII visual, then a summary. Follow-up questions ("why is it O(n log n)?") are detected separately and rendered as normal chat replies instead of forcing the rigid structure onto every message.

## Features

- **Structured explanations**: theory → example → step-by-step (with ASCII diagrams) → summary, parsed from the LLM's XML-tagged output and rendered as an animated card (steps reveal one at a time).
- **Conversational follow-ups**: once an algorithm's been explained, you can ask clarifying questions and get plain conversational answers — no rigid format.
- **Multi-turn memory**: full conversation history is sent with each request, so follow-ups have context.
- **Custom XML-response parsing**: a lightweight regex-based parser (no external XML library) extracts `<theory>`, `<steps>`, `<step>`, `<visual>`, `<summary>` etc. from the raw model output.
- **Single-page vanilla JS frontend**: no framework, no build step — auto-resizing input, typing indicator, animated step reveal, all in one HTML file.

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | Flask (Python) |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Frontend | Vanilla JS, HTML, CSS (no framework) |
| Config | `python-dotenv` for API key management |
| Server | `gunicorn` (production) |

## How it works

1. User sends a message (either "Explain X" or a follow-up question).
2. Flask forwards it to Groq's chat completions endpoint along with a system prompt that instructs the model to detect whether this is an **explain request** or a **follow-up/doubt**, and respond accordingly (structured XML vs. plain text).
3. The backend parses the raw response:
   - If it contains `<theory>` and `<steps>` tags → treated as a full explanation, parsed into a JSON object (`theory`, `example_intro`, `steps[]`, `summary`).
   - Otherwise → treated as a follow-up and passed through as plain text.
4. The frontend renders either an animated **explanation card** (steps fade in sequentially) or a normal **chat bubble**, and appends the raw model output to conversation history for the next turn.

## Project Structure

```
.
├── app.py          # Flask backend: routes, Groq API call, XML parsing
├── index.html       # Single-page frontend: chat UI, rendering, animations
├── requirements.txt
└── .env             # GROQ_API_KEY (not committed)
```

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the root with your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open `http://localhost:5000` in your browser.

## Example

**User:** `Explain Binary Search`
**Bot:** Renders a card with theory + complexity, a worked example intro, 4+ steps each with an ASCII array visual (e.g. `[1, 3, 5, 8, 9] -> mid=5`) and explanation, and a summary of key takeaways.

**User (follow-up):** `What if the array isn't sorted?`
**Bot:** Replies conversationally, explaining binary search's precondition, without re-triggering the structured card format.

## Design Decisions

- **XML tags over JSON for LLM output**: LLMs are generally more reliable at producing well-formed custom tags than strictly valid JSON, especially with free-form prose (like `theory` or `explanation` fields) mixed in. Regex extraction is simple and avoids adding an XML parsing dependency.
- **Two response modes from one prompt**: rather than running two separate prompts/classifiers, the system prompt asks the model to self-detect intent (explain vs. follow-up) and switch output format itself, keeping the request flow simple.
- **Client-side conversation history**: history is kept in browser memory and sent with each request rather than persisted server-side, since this is a lightweight single-session tool.

## Possible Improvements

- Persist conversation history (e.g. SQLite) so sessions survive a page refresh.
- Add a "regenerate" option per explanation.
- Support code-based visualizations (e.g. actual sorting animations) alongside ASCII.
- Rate-limit / cache repeated identical explain requests to reduce API calls.
