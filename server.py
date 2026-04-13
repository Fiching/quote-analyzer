"""
Quote Analyzer — Flask + Claude API backend
Deploy to Render.com

Environment variables:
    ANTHROPIC_API_KEY
    APP_PASSWORD
    SECRET_KEY
"""

from flask import (
    Flask, request, jsonify, session,
    send_from_directory, Response, stream_with_context, redirect
)
import anthropic
import os, functools, json, re
from datetime import timedelta

PASSWORD   = os.environ.get("APP_PASSWORD", "changeme123")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL      = "claude-haiku-4-5-20251001"
PORT       = int(os.environ.get("PORT", 5000))

app = Flask(__name__, static_folder=".", template_folder=".")
app.secret_key = SECRET_KEY
app.permanent_session_lifetime = timedelta(days=30)

# ── SIZE CODE MAP ─────────────────────────────────────────────────────────────
SIZE = {
    'A':'1/8','B':'1/4','C':'3/8','D':'1/2','F':'3/4',
    'G':'1','H':'1-1/4','J':'1-1/2','K':'2','L':'2-1/2',
    'M':'3','N':'3-1/2','P':'4','R':'4-1/2','S':'5','T':'5-1/2','U':'6'
}
SIZE_REV = {v:k for k,v in SIZE.items()}

# ── PART NUMBER DECODER ───────────────────────────────────────────────────────
def decode_part(pn):
    """
    Decode a part number into its components.
    Returns dict with: family, material, connection, fitting_type, sizes, valid, issues
    """
    pn = pn.strip().upper()
    result = {'pn': pn, 'family': None, 'material': None, 'connection': None,
              'fitting_type': None, 'sizes': [], 'valid': False, 'issues': [], 'decoded_desc': None}

    # ── FST / FSS / FS6 ──────────────────────────────────────────────────────
    if pn.startswith('FST') or pn.startswith('FSS') or pn.startswith('FS6'):
        result['material'] = 'FS'  # Forged Steel
        if pn.startswith('FST'): result['connection'] = 'THRD'; body = pn[3:]
        elif pn.startswith('FSS'): result['connection'] = 'SW'; body = pn[3:]
        else: result['connection'] = 'SW 6000#'; body = pn[3:]

        # Fitting type prefixes (longest first)
        fit_map = [
            ('CAP','CAP'),('SSU','UNION'),('HC','HALF COUP'),('HP','HEX PLUG'),
            ('SHP','SQ HD PLUG'),('RHP','RND HD PLUG'),('S9','ST 90 ELL'),
            ('9','90 ELL'),('4','45 ELL'),('T','TEE'),('C','COUP'),
            ('B','BUSHING'),('R','RED COUP'),('I','INSERT'),
        ]
        matched_type = None
        for prefix, label in fit_map:
            if body.startswith(prefix):
                matched_type = label
                body = body[len(prefix):]
                break

        if not matched_type:
            result['issues'].append(f'Unknown fitting type in {pn}')
            return result

        result['fitting_type'] = matched_type

        # Parse size codes
        sizes = []
        i = 0
        while i < len(body):
            # Check two-char codes first
            if body[i:i+2] in SIZE_REV or body[i:i+2] == 'CL':
                sizes.append(body[i:i+2]); i += 2
            elif body[i] in SIZE:
                sizes.append(body[i]); i += 1
            else:
                result['issues'].append(f'Unknown size code: {body[i:]} in {pn}')
                break

        if matched_type in ('BUSHING','RED COUP','INSERT') and len(sizes) < 2:
            result['issues'].append(f'{matched_type} requires 2 sizes, only found {len(sizes)}')
        elif matched_type not in ('BUSHING','RED COUP','INSERT') and len(sizes) < 1:
            result['issues'].append(f'{matched_type} requires a size code')

        result['sizes'] = [SIZE.get(s, s) for s in sizes]
        result['family'] = 'FST' if pn.startswith('FST') else 'FSS'
        result['valid'] = len(result['issues']) == 0

        # Build decoded description
        if result['valid']:
            if matched_type in ('BUSHING','RED COUP') and len(result['sizes']) >= 2:
                size_str = f"{result['sizes'][0]}X{result['sizes'][1]}"
            elif result['sizes']:
                size_str = result['sizes'][0]
            else:
                size_str = ''
            rating = '6000#' if 'FS6' in pn[:3] else '3000#'
            result['decoded_desc'] = f"{size_str} FS {rating} {result['connection']} {matched_type}"
        return result

    # ── BXSN (Black XH Seamless Nipple) ──────────────────────────────────────
    if pn.startswith('BXSN'):
        result['family'] = 'BXSN'
        result['material'] = 'BLK XH'
        result['fitting_type'] = 'NIP'
        result['connection'] = 'SMLS A106'
        body = pn[4:]
        if not body:
            result['issues'].append('Missing size code')
            return result
        size_char = body[0]
        if size_char not in SIZE:
            result['issues'].append(f'Unknown size: {size_char}')
            return result
        pipe_size = SIZE[size_char]
        length_code = body[1:]
        if not length_code:
            result['issues'].append('Missing length (e.g. M=3", CL=close)')
            return result
        length = 'CLOSE' if length_code == 'CL' else SIZE.get(length_code, f'?({length_code})')
        result['sizes'] = [pipe_size, length]
        result['valid'] = True
        result['decoded_desc'] = f"{pipe_size}X{length} BLK XH SMLS A106 NIP"
        return result

    # ── GXSN (Galvanized XH Seamless Nipple) ─────────────────────────────────
    if pn.startswith('GXSN'):
        result['family'] = 'GXSN'
        result['material'] = 'GALV XH'
        result['fitting_type'] = 'NIP'
        body = pn[4:]
        if not body or body[0] not in SIZE:
            result['issues'].append('Missing size code')
            return result
        pipe_size = SIZE[body[0]]
        length_code = body[1:]
        length = 'CLOSE' if length_code == 'CL' else SIZE.get(length_code, f'?({length_code})')
        result['sizes'] = [pipe_size, length]
        result['valid'] = True
        result['decoded_desc'] = f"{pipe_size}X{length} GALV XH SMLS A106 NIP"
        return result

    # ── GRFB/GRFW/GRFS/GRFT (Domestic CS Flanges) ────────────────────────────
    if pn.startswith('GRF'):
        result['family'] = 'GRFB'
        result['material'] = 'CS'
        result['fitting_type'] = 'FLANGE'
        flg_types = {'B':'BLND','W':'WN','S':'SO','T':'THRD'}
        if len(pn) > 3 and pn[3] in flg_types:
            result['fitting_type'] = f"{flg_types[pn[3]]} FLG"
        result['valid'] = True
        return result

    # ── GGLFB etc (Global/Import Flanges) ─────────────────────────────────────
    if pn.startswith('GGL') or pn.startswith('GGR'):
        result['family'] = 'IMPORT_FLG'
        result['material'] = 'CS GLOBAL'
        result['fitting_type'] = 'FLANGE'
        result['valid'] = True
        return result

    # ── IS (Import Stainless) ─────────────────────────────────────────────────
    if pn.startswith('IS4') or pn.startswith('IS6') or pn.startswith('IS3'):
        result['family'] = 'IS'
        result['material'] = 'SS IMPORT'
        result['fitting_type'] = 'FITTING'
        result['valid'] = True
        return result

    # ── Unknown ───────────────────────────────────────────────────────────────
    result['issues'].append(f'Unknown part number family: {pn}')
    return result


# ── CATALOG LOADER ────────────────────────────────────────────────────────────
_catalog_cache = None

def load_catalog_lookup():
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache
    path = os.path.join(os.path.dirname(__file__), "catalog.txt")
    lookup = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    lookup[parts[0].strip()] = parts[1].strip()
    _catalog_cache = lookup
    return lookup


def load_file(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(path): return ""
    with open(path, 'r', encoding='utf-8') as f: return f.read().strip()


def load_patterns():
    raw = load_file("patterns.txt")
    lines = [l for l in raw.splitlines() if not l.strip().startswith('#')]
    return '\n'.join(lines).strip()


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

@app.route("/api/validate", methods=["POST"])
def validate():
    """Validate and decode a list of part numbers."""
    data = request.get_json(silent=True) or {}
    parts = data.get("parts", [])
    lookup = load_catalog_lookup()
    results = []
    for pn in parts:
        pn = pn.strip().upper()
        in_catalog = pn in lookup
        decoded = decode_part(pn)
        results.append({
            "pn": pn,
            "in_catalog": in_catalog,
            "description": lookup.get(pn, ""),
            "decoded": decoded
        })
    return jsonify({"results": results})

@app.route("/api/analyze", methods=["POST"])
@login_required
def analyze():
    if not API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    data = request.get_json(silent=True) or {}
    quote = data.get("quote", "").strip()
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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


if __name__ == "__main__":
    lookup = load_catalog_lookup()
    print(f"\n  Quote Analyzer running on http://0.0.0.0:{PORT}")
    print(f"  Catalog: {len(lookup)} parts")
    print(f"  Model:   {MODEL}")
    print(f"  API key: {'set' if API_KEY else 'NOT SET'}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
