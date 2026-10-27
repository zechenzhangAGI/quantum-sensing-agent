"""Shared log parser for the QSA agent logs (read-only on the repo).

A log entry starts with "[YYYY-MM-DD HH:MM:SS] ROLE: ..." at the start of a line; any following lines
without that prefix (multi-line LLM output, script stdout) belong to the same entry.
"""
import os, re, glob, json
from datetime import datetime

# QSA_REPO selects the checkout (default: this repository). The two Feb 2025 root logs live under runs/.
# Inside the released repository (analysis/ next to agent/) the repository itself is used.
_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.expanduser(os.environ.get('QSA_REPO') or _HERE)
# Files of the 14 May 2025 session (run_20250514_082859) used by esr_fit.py and make_figures.py.
CASE_DIR = os.path.expanduser(os.environ.get('QSA_CASE_DIR') or f'{REPO}/examples/2025-05-14-nv-odmr/data')
RUNS = f'{REPO}/projects/NVExperiment/runs'
EMB = f'{REPO}/projects/NVExperiment/embeddings'

# The copies of the two root logs
# under runs/ are byte-identical (checked with cmp) and are not counted twice.
def log_files():
    fs = [f'{REPO}/{n}' if os.path.exists(f'{REPO}/{n}') else f'{RUNS}/{n}' for n in ('agent_history.log', 'agent_history_new.log')]
    fs += sorted(glob.glob(f'{RUNS}/run_*/logs/*.log'))
    return fs

def short(f):
    # Root logs are reported under their root name whichever copy was read.
    return f.replace(REPO + '/', '').replace('projects/NVExperiment/runs/agent_history', 'agent_history')

HEAD = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:,\d+)?\] ([A-Z_]+): ?(.*)$')

def parse(f):
    """Return list of entries: dict(ts, role, text, line) with line = 1-based start line."""
    out = []
    # Split on '\n' only: some logs contain bare '\r' (Windows script stdout), and line numbers must match grep -n.
    with open(f, encoding='utf-8', errors='replace', newline='') as fh:
        lines = fh.read().split('\n')
    for i, line in enumerate(lines, 1):
        line = line.rstrip('\r')
        m = HEAD.match(line)
        if m:
            out.append(dict(ts=datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S'), role=m.group(2),
                            text=m.group(3), line=i))
        elif out:
            out[-1]['text'] += '\n' + line
    return out

# A completed measurement is a "[Runner] Saved <type> data to:" line
# at the start of a line of logged script stdout.
SAVED = re.compile(r'^\[Runner\] Saved (GalvoScan|FindNV|ESR|esr|optimization|Optimization) data to: (.*)$', re.M)

def mtype(k):
    return 'ESR' if k.lower() == 'esr' else 'Optimize' if k.lower() == 'optimization' else k

HW = re.compile(r'PB connected|\[Runner\] Saved')

ACK = re.compile(r"^(y|yes|yes+|ok|okay|sure|proceed|go ahead|ok go ahead|ok proceed|yes proceed|yes please|yes you may|"
                 r"please go ahead and run|run it|ok good|ok do the measurement|(yes)+)[.!]?$", re.I)

def classify_user(text):
    t = text.strip()
    if t.startswith('(permission)'):
        ans = t[len('(permission)'):].strip().lower()
        return 'perm_yes' if ans in ('yes', 'y') else 'perm_no' if ans in ('no', 'n') else 'perm_other'
    if ACK.match(t):
        return 'ack'
    return 'guidance'

ACTION_KINDS = [
    ('MESSAGE', r'^MESSAGE: '), ('READ', r'^READ: '), ('WRITE', r'^WRITE file with content: '),
    ('RUN', r'^RUN: '), ('VISION', r'^VISION: '), ('RUN_OUTPUT', r'^RUN OUTPUT: '),
    ('WRITE_DENIED', r'^WRITE DENIED for '), ('RUN_DENIED', r'^RUN DENIED for '), ('VISION_DENIED', r'^VISION DENIED for '),
    ('ASK_PERMISSION', r'^\(ASK PERMISSION\) '), ('RUNNING', r'^Running command: '),
    ('MODIFIED_CMD', r'^Modified command with output directory: '), ('SYSTEM', r'^\[System\] '),
]

def action_kind(text):
    for k, p in ACTION_KINDS:
        if re.match(p, text):
            return k
    return 'OTHER'

# <action>{ "type": ..., "content": ...}</action> blocks in raw LLM output. JSON is often loose, so take the type
# with a regex instead of json.loads.
ATYPE = re.compile(r'<action>\s*\{\s*"type"\s*:\s*"([a-zA-Z_]+)"', re.S)

def proposed_actions(text):
    return [t.lower() for t in ATYPE.findall(text)]

def is_llm_turn(e):
    # Raw LLM replies; "(THINK)" / June auto-mode "(THINK-AUTO-n)" entries are the extracted reasoning of the same reply.
    return e['role'] == 'ASSISTANT' and not e['text'].startswith('(THINK')

def segments(entries, gap_h=3.0):
    """Split a log into sittings separated by > gap_h hours of silence."""
    segs, cur = [], []
    for e in entries:
        if cur and (e['ts'] - cur[-1]['ts']).total_seconds() > gap_h * 3600:
            segs.append(cur); cur = []
        cur.append(e)
    if cur:
        segs.append(cur)
    return segs
