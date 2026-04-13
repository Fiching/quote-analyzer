"""
Quote Analyzer — Flask + Claude API backend
Deploy to Render.com

Environment variables required (set in Render dashboard):
    ANTHROPIC_API_KEY   — your Claude API key
    APP_PASSWORD        — login password for the app
    SECRET_KEY          — random secret for sessions (generate any long random string)
"""

from flask import (
    Flask, request, jsonify, session,
    send_from_directory, Response, stream_with_context, redirect
)
import anthropic
import os
import functools
import json
from datetime import timedelta

# ── CONFIG ────────────────────────────────────────────────────────────────────
PASSWORD   = os.environ.get("APP_PASSWORD", "changeme123")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-haiku-4-5-20251001"  
PORT       = int(os.environ.get("PORT", 5000))
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=".", template_folder=".")
app.secret_key = SECRET_KEY
app.permanent_session_lifetime = timedelta(days=30)


def load_file(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_patterns():
    raw = load_file("patterns.txt")
    lines = [l for l in raw.splitlines() if not l.strip().startswith("#")]
    return "\n".join(lines).strip()


def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authed"):
            return jsonify({"error": "unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if not session.get("authed"):
        return send_from_directory(".", "login.html")
    return send_from_directory(".", "index.html")


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    if data.get("password") == PASSWORD:
        session.permanent = True
        session["authed"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Wrong password"}), 401


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect("/")


@app.route("/api/catalog")
def catalog():
    return jsonify({"catalog": load_file("catalog.txt")})


@app.route("/api/patterns")
def patterns():
    return jsonify({"patterns": load_patterns()})


@app.route("/api/debug")
def debug():
    key = API_KEY
    masked = key[:12] + "..." + key[-4:] if len(key) > 16 else "NOT SET"
    return jsonify({"api_key_set": bool(API_KEY), "api_key_preview": masked, "model": MODEL})


@app.route("/api/analyze", methods=["POST"])
@login_required
def analyze():
    if not API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    data = request.get_json(silent=True) or {}
    quote = data.get("quote", "").strip()
    examples = data.get("examples", [])
    system_prompt = data.get("system_prompt", "")

    if not quote:
        return jsonify({"error": "no quote provided"}), 400

    client = anthropic.Anthropic(api_key=API_KEY)

    def generate():
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": quote}]
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    catalog_data = load_file("catalog.txt")
    part_count = len([l for l in catalog_data.split("\n") if l.strip()])
    print(f"\n  Quote Analyzer running on http://0.0.0.0:{PORT}")
    print(f"  Catalog:   {part_count} parts")
    print(f"  Model:     {MODEL}")
    print(f"  API key:   {'set' if API_KEY else 'NOT SET — add ANTHROPIC_API_KEY'}")
    print(f"\n  Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
