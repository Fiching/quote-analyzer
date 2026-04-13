"""
Quote Analyzer — Flask + Claude API
Features: Tool Use, Prompt Caching, File Upload, Claude Validation
"""

from flask import (
    Flask, request, jsonify, session,
    send_from_directory, Response, stream_with_context, redirect
)
import anthropic
import os, functools, json, re, base64
from datetime import timedelta

PASSWORD   = os.environ.get("APP_PASSWORD", "changeme123")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL      = "claude-haiku-4-5-20251001"
PORT       = int(os.environ.get("PORT", 5000))

app = Flask(__name__, static_folder=".", template_folder=".")
app.secret_key = SECRET_KEY
app.permanent_session_lifetime = timedelta(days=30)

# ── CATALOG ───────────────────────────────────────────────────────────────────
_catalog = None

def get_catalog():
    global _catalog
    if _catalog is not None:
        return _catalog
    path = os.path.join(os.path.dirname(__file__), "catalog.txt")
    parts = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                bits = line.strip().split('\t')
                if len(bits) >= 2:
                    pn = bits[0].strip()
                    desc = bits[1].strip()
                    tokens = set(re.sub(r'[^A-Z0-9/]', ' ', desc.upper()).split())
                    parts.append({'pn': pn, 'desc': desc, 'tokens': tokens})
    _catalog = parts
    return parts

def get_catalog_text():
    path = os.path.join(os.path.dirname(__file__), "catalog.txt")
    if not os.path.exists(path): return ""
    with open(path, 'r', encoding='utf-8') as f: return f.read().strip()

def get_patterns():
    path = os.path.join(os.path.dirname(__file__), "patterns.txt")
    if not os.path.exists(path): return ""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l for l in f.read().splitlines() if not l.strip().startswith('#')]
        return '\n'.join(lines).strip()

# ── PART NUMBER DECODER ───────────────────────────────────────────────────────
SIZE = {
    'A':'1/8','B':'1/4','C':'3/8','D':'1/2','F':'3/4',
    'G':'1','H':'1-1/4','J':'1-1/2','K':'2','L':'2-1/2',
    'M':'3','N':'3-1/2','P':'4','R':'4-1/2','S':'5','T':'5-1/2','U':'6'
}

def decode_part(pn):
    pn = pn.strip().upper()
    result = {'pn':pn,'valid':False,'issues':[],'family':None,'decoded_desc':None}
    cats = get_catalog()
    in_catalog = any(p['pn'] == pn for p in cats)
    result['in_catalog'] = in_catalog

    if pn.startswith('FST') or pn.startswith('FSS') or pn.startswith('FS6'):
        conn = 'THRD' if pn.startswith('FST') else 'SW'
        rating = '6000#' if pn.startswith('FS6') else '3000#'
        body = pn[3:]
        fit_map = [('CAP','CAP'),('SSU','UNION'),('HC','HALF COUP'),
                   ('HP','HEX PLUG'),('S9','ST 90 ELL'),('9','90 ELL'),
                   ('4','45 ELL'),('T','TEE'),('C','COUP'),
                   ('B','BUSHING'),('R','RED COUP')]
        fit_label = None
        for prefix, label in fit_map:
            if body.startswith(prefix):
                fit_label = label
                body = body[len(prefix):]
                break
        if not fit_label:
            result['issues'].append(f'Unknown fitting type')
            return result
        sizes = []
        i = 0
        while i < len(body):
            if body[i:i+2] == 'CL': sizes.append('CLOSE'); i+=2
            elif body[i] in SIZE: sizes.append(SIZE[body[i]]); i+=1
            else: break
        if fit_label in ('BUSHING','RED COUP') and len(sizes) < 2:
            result['issues'].append(f'{fit_label} needs 2 sizes')
        size_str = 'x'.join(sizes) if sizes else '?'
        result['decoded_desc'] = f"{size_str} FS {rating} {conn} {fit_label}"
        result['valid'] = len(result['issues']) == 0
        result['family'] = pn[:3]
        return result

    if pn.startswith('BXSN') or pn.startswith('GXSN'):
        mat = 'BLK XH' if pn.startswith('BX') else 'GALV XH'
        body = pn[4:]
        if not body or body[0] not in SIZE:
            result['issues'].append('Missing size')
            return result
        pipe_size = SIZE[body[0]]
        length_code = body[1:]
        length = 'CLOSE' if length_code == 'CL' else SIZE.get(length_code, f'?')
        result['decoded_desc'] = f"{pipe_size}x{length} {mat} SMLS A106 NIP"
        result['valid'] = True
        result['family'] = pn[:4]
        return result

    result['valid'] = in_catalog
    return result

# ── TOOL: lookup_part ─────────────────────────────────────────────────────────
def lookup_part(description, size=None, fitting_type=None, connection=None, material=None):
    """Search catalog and return best matches."""
    parts = get_catalog()
    query_tokens = set(re.sub(r'[^A-Z0-9/]', ' ', description.upper()).split())
    if size: query_tokens.update(re.sub(r'[^A-Z0-9/]', ' ', size.upper()).split())
    if fitting_type: query_tokens.update(fitting_type.upper().split())
    if connection: query_tokens.add(connection.upper())
    if material: query_tokens.update(material.upper().split())

    scored = []
    for p in parts:
        score = len(query_tokens & p['tokens'])
        if score > 0:
            scored.append((score, p['pn'], p['desc']))

    scored.sort(key=lambda x: -x[0])
    top = scored[:5]

    if not top:
        return {"found": False, "message": "No matching parts found", "matches": []}

    exact = [m for m in top if m[0] >= len(query_tokens) * 0.8]
    return {
        "found": True,
        "exact_match": len(exact) > 0,
        "matches": [{"part_number": m[1], "description": m[2], "score": m[0]} for m in top]
    }

TOOLS = [{
    "name": "lookup_part",
    "description": (
        "Search the parts catalog to find matching part numbers. "
        "Call this for EACH line item in the quote. "
        "Use the best match if confidence is high, flag as ESTIMATED if unsure."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Full description of the part"},
            "size": {"type": "string", "description": "Pipe size e.g. '1/2', '3/4x1/2'"},
            "fitting_type": {"type": "string", "description": "e.g. '90 ELL', 'TEE', 'BUSHING', 'NIPPLE'"},
            "connection": {"type": "string", "description": "THRD or SW"},
            "material": {"type": "string", "description": "e.g. 'BLK XH', 'GALV', 'FS', 'SS'"}
        },
        "required": ["description"]
    }
}]

SYSTEM_PROMPT = """You are a specialist parts quote analyzer for an industrial pipe and fitting supplier.

For EACH line item in the quote, you MUST call the lookup_part tool to search the catalog.

OUTPUT FORMAT — after all tool calls, return ONLY a pipe-delimited table:
QTY|PART_NUMBER|DESCRIPTION

Rules:
- Call lookup_part for every single line item — never skip
- If lookup returns an exact match with high confidence: use that part number
- If lookup returns a close but uncertain match: prefix with EST: (e.g. EST:FST9K)
- If required info is missing (e.g. bushing needs 2 sizes, elbow needs THRD or SW): output QTY|NEED MORE INFO|what is missing
- If truly no match found: output QTY|NOT FOUND|customer description
- Default quantity to 1 if not specified
- Default rating to 3000# for FS fittings unless stated otherwise
- Do NOT assume THRD vs SW — always flag as NEED MORE INFO if not specified
- Bushing always needs two sizes — if only one given, flag NEED MORE INFO

PART NUMBER STRUCTURE:
FST=Forged Steel Threaded 3000# | FSS=Forged Steel Socket Weld 3000# | FS6=FS SW 6000#
Size codes: A=1/8 B=1/4 C=3/8 D=1/2 F=3/4 G=1 H=1-1/4 J=1-1/2 K=2 L=2-1/2 M=3 N=3-1/2 P=4 U=6
Fitting: 9=90ELL 4=45ELL T=TEE C=COUP B=BUSH R=REDCOUP CAP=CAP SSU=UNION HC=HALFCOUP
BXSN=Black XH Seamless Nipple | GXSN=Galv XH Seamless Nipple
Nipple: BXSN[size][length] e.g. BXSNDM=1/2x3" | BXSNDCL=1/2 CLOSE
Bushing/RedCoup: always TWO size codes e.g. FSTBFD=3/4x1/2

MISSING INFO RULES:
- FS fitting with no THRD/SW specified → NEED MORE INFO: Threaded or socket weld?
- Bushing with only one size → NEED MORE INFO: What is the reduction size?
- Nipple with no length → NEED MORE INFO: What length? (or CLOSE?)
- Any fitting with no size → NEED MORE INFO: What pipe size?

Output nothing except the table. No explanation, no preamble."""

# ── AUTH ──────────────────────────────────────────────────────────────────────
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
    if not session.get("authed"): return send_from_directory(".", "login.html")
    return send_from_directory(".", "index.html")

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    if data.get("password") == PASSWORD:
        session.permanent = True; session["authed"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Wrong password"}), 401

@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear(); return redirect("/")

@app.route("/api/debug")
def debug():
    key = API_KEY
    masked = key[:12] + "..." + key[-4:] if len(key) > 16 else "NOT SET"
    cats = get_catalog()
    return jsonify({"api_key_set": bool(API_KEY), "api_key_preview": masked,
                    "model": MODEL, "catalog_size": len(cats)})

@app.route("/api/validate", methods=["POST"])
def validate():
    data = request.get_json(silent=True) or {}
    parts = data.get("parts", [])
    results = []
    for pn in parts:
        decoded = decode_part(pn.strip().upper())
        results.append({"pn": pn, "in_catalog": decoded.get("in_catalog", False), "decoded": decoded})
    return jsonify({"results": results})

@app.route("/api/analyze", methods=["POST"])
@login_required
def analyze():
    if not API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    data = request.get_json(silent=True) or {}
    quote = data.get("quote", "").strip()
    examples = data.get("examples", [])
    file_data = data.get("file_data")    # base64 encoded
    file_type = data.get("file_type")    # image/jpeg, image/png, application/pdf

    if not quote and not file_data:
        return jsonify({"error": "no input provided"}), 400

    client = anthropic.Anthropic(api_key=API_KEY)

    # Build system prompt with caching
    sys_prompt_parts = [{"type": "text", "text": SYSTEM_PROMPT}]

    # Add training examples if any
    if examples:
        ex_text = "\n\nTRAINING EXAMPLES — follow this exact logic:\n"
        for i, ex in enumerate(examples):
            ex_text += f"\nExample {i+1}:\nInput: {ex['quote']}\nCorrect output:\n{ex['answer']}\n"
        sys_prompt_parts.append({
            "type": "text",
            "text": ex_text,
            "cache_control": {"type": "ephemeral"}
        })

    # Build user message
    if file_data and file_type:
        if file_type == "application/pdf":
            user_content = [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": file_data}},
                {"type": "text", "text": quote or "Analyze this quote request and output the parts table."}
            ]
        else:
            user_content = [
                {"type": "image", "source": {"type": "base64", "media_type": file_type, "data": file_data}},
                {"type": "text", "text": quote or "Analyze this quote request and output the parts table."}
            ]
    else:
        user_content = quote

    messages = [{"role": "user", "content": user_content}]

    def generate():
        try:
            # Agentic tool use loop
            full_text = ""
            while True:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=2048,
                    system=sys_prompt_parts,
                    tools=TOOLS,
                    messages=messages
                )

                # Process tool calls
                tool_calls_made = False
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        tool_calls_made = True
                        tool_input = block.input
                        result = lookup_part(
                            description=tool_input.get("description", ""),
                            size=tool_input.get("size"),
                            fitting_type=tool_input.get("fitting_type"),
                            connection=tool_input.get("connection"),
                            material=tool_input.get("material")
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result)
                        })
                    elif block.type == "text":
                        full_text += block.text

                if tool_calls_made:
                    # Add assistant response and tool results to messages
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
                    # Stream progress indicator
                    yield f"data: {json.dumps({'text': ''})}\n\n"
                else:
                    # No more tool calls — stream the final text
                    for char in full_text:
                        yield f"data: {json.dumps({'text': char})}\n\n"
                    break

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

if __name__ == "__main__":
    cats = get_catalog()
    print(f"\n  Quote Analyzer")
    print(f"  Catalog: {len(cats)} parts")
    print(f"  Model:   {MODEL}")
    print(f"  API key: {'set' if API_KEY else 'NOT SET'}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
