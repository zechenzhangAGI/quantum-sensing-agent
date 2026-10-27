"""Derive every log-based number in the technical report. Writes stats.json and data/*.csv.

Run: .venv/bin/python -I scripts/analyze.py   (from the analysis/ folder)
"""
import os, re, csv, json, sys, statistics as st
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qsa_logs import *

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f'{OUT}/data'
os.makedirs(DATA, exist_ok=True)

def version(f, s):
    """Agent version that wrote a log, from log-format markers and date."""
    d = re.search(r'(\d{8})_\d{6}', f)
    d = d.group(1) if d else '20250218'
    if d >= '20250601':
        return 'v4 June code, assistant/auto modes + rag_search (origin/june-14-system-prompt-fixes)'
    if 'Building prompt with RAG context' in s:
        return 'v2 RAG prototype (2933be4, Jack Fan, dev machine)' if '] RAG: ' in s else \
               'v3 lab version, RAG prompt + embeddings on exit (closest commit 7943e46)'
    if 'Modified command with output directory' in s:
        return 'v1b uncommitted local variant (19 Apr)'
    if d <= '20250314':
        return 'v0 pre-repo (code as imported in 17c17e1)'
    return 'v1 Apr refactor (162a5d5 / 86e2e52, Jack Fan)'

def script_of(cmd):
    m = re.search(r'(galvo_scan|find_nv|optimize|ESR)\.py', cmd, re.I)
    return m.group(1) if m else None

sessions, meas, runs, failures, guidance, perms = [], [], [], [], [], []
act_exec, act_prop = Counter(), Counter()
act_exec_by_ver, act_prop_by_ver = defaultdict(Counter), defaultdict(Counter)
llm_latency, human_latency = [], []
# LLM responses that continue the dialogue on their own: lines starting "User:" / "Assistant:" inside one raw
# response, i.e. invented user replies and invented tool output. Their "[Runner] Saved" lines are not measurements.
FAKE_TURN = re.compile(r'^(User|Assistant): ', re.M)
hallucinated = []

for f in log_files():
    E = parse(f)
    name = short(f)
    s_all = open(f, encoding='utf-8', errors='replace').read()
    ver = version(name, s_all)
    for si, seg in enumerate(segments(E)):
        c = Counter()
        n_llm = sum(is_llm_turn(e) for e in seg)
        uc = Counter(classify_user(e['text']) for e in seg if e['role'] == 'USER')
        hw = any(e['role'] == 'ACTION' and action_kind(e['text']) == 'RUN_OUTPUT' and HW.search(e['text']) for e in seg)
        last_run = None
        for i, e in enumerate(seg):
            if is_llm_turn(e):
                if FAKE_TURN.search(e['text']):
                    hallucinated.append(dict(log=name, line=e['line'], ts=str(e['ts']), version=ver,
                                             fake_user_lines=len(re.findall(r'^User: ', e['text'], re.M)),
                                             fake_saved_lines=len(SAVED.findall(e['text']))))
                pa = proposed_actions(e['text'])
                act_prop.update(pa); act_prop_by_ver[ver].update(pa)
                # LLM latency: previous entry is the USER turn that triggered this call (or AGENT/RAG lines after it)
                j = i - 1
                while j >= 0 and seg[j]['role'] in ('AGENT', 'RAG'):
                    j -= 1
                if j >= 0 and seg[j]['role'] == 'USER':
                    llm_latency.append((e['ts'] - seg[j]['ts']).total_seconds())
            if e['role'] == 'USER':
                k = classify_user(e['text'])
                # human latency: time since the last logged agent entry
                if i > 0 and seg[i - 1]['role'] in ('ACTION', 'ASSISTANT'):
                    human_latency.append((e['ts'] - seg[i - 1]['ts']).total_seconds())
                if k.startswith('perm'):
                    turn = next((x for x in reversed(seg[:i]) if is_llm_turn(x)), None)
                    perms.append(dict(log=name, line=e['line'], ts=str(e['ts']), answer=k, version=ver,
                                      llm_turn_line=turn['line'] if turn else '',
                                      llm_turn_hallucinated=bool(turn and FAKE_TURN.search(turn['text']))))
                if k == 'guidance':
                    nxt = next((x for x in seg[i + 1:] if x['role'] == 'ASSISTANT' and x['text'].startswith('(THINK)')), None)
                    guidance.append(dict(log=name, line=e['line'], ts=str(e['ts']), hw_session=hw, version=ver,
                                         text=e['text'].strip().replace('\n', ' ')[:300],
                                         next_think_line=nxt['line'] if nxt else '',
                                         next_think=(nxt['text'][8:].strip().replace('\n', ' ')[:300] if nxt else '')))
            if e['role'] != 'ACTION':
                continue
            k = action_kind(e['text'])
            base = {'WRITE_DENIED': 'WRITE', 'RUN_DENIED': 'RUN', 'VISION_DENIED': 'VISION'}.get(k, k)
            if base in ('MESSAGE', 'READ', 'WRITE', 'RUN', 'VISION'):
                key = base.lower() + ('_denied' if k.endswith('DENIED') else '')
                act_exec[key] += 1; act_exec_by_ver[ver][key] += 1; c[key] += 1
            if k == 'VISION':
                c['vision'] += 0
            if k == 'RUN':
                last_run = e
                runs.append(dict(log=name, line=e['line'], ts=str(e['ts']), script=script_of(e['text']),
                                 cmd=e['text'][5:].split('\n')[0][:200]))
            if k == 'RUN_OUTPUT':
                saved = [(mtype(m.group(1)), m.group(2).strip()) for m in SAVED.finditer(e['text'])]
                dur = (e['ts'] - last_run['ts']).total_seconds() if last_run else None
                runs[-1]['outcome'] = 'output' + ('+saved' if saved else '')
                for t, p in saved:
                    meas.append(dict(log=name, line=e['line'], date=str(e['ts'].date()), ts_end=str(e['ts']),
                                     ts_start=str(last_run['ts']) if last_run else '', type=t,
                                     script_wall_s=dur, file=os.path.basename(p.replace('\\', '/'))))
            if k == 'SYSTEM' and re.match(r'\[System\] (Error running command|READ denied|WRITE denied|File not found)', e['text']):
                msg = e['text'].split('\n')[0]
                cat = ('exit status ' + re.search(r'exit status (\d+)', msg).group(1)) if 'exit status' in msg else \
                      'command not allowed' if 'not allowed' in msg else 'command parse error' if 'parsing error' in msg else \
                      'read path denied' if 'READ denied' in msg else 'write path denied' if 'WRITE denied' in msg else \
                      'file not found' if 'File not found' in msg else 'other'
                if 'Error running command' in msg and runs:
                    runs[-1]['outcome'] = cat
                failures.append(dict(log=name, line=e['line'], ts=str(e['ts']), category=cat, hw_session=hw,
                                     msg=msg[9:200]))
            if k == 'RUN_OUTPUT':
                # script ran but wrote a traceback / KeyError (bad config keys) to its output
                if re.search(r'Traceback|KeyError|Error:', e['text']) and not SAVED.search(e['text']):
                    failures.append(dict(log=name, line=e['line'], ts=str(e['ts']), category='script error in output',
                                         hw_session=hw, msg=re.findall(r'(\w*Error[^\n]{0,150})', e['text'])[-1:]))
        mc = Counter()
        for e in seg:
            if e['role'] == 'ACTION':
                for m in SAVED.finditer(e['text']):
                    mc[mtype(m.group(1))] += 1
        sessions.append(dict(
            log=name, sitting=si + 1, date=str(seg[0]['ts'].date()), start=str(seg[0]['ts']), end=str(seg[-1]['ts']),
            duration_min=round((seg[-1]['ts'] - seg[0]['ts']).total_seconds() / 60, 1), version=ver,
            reached_hardware=hw, llm_turns=n_llm, user_msgs=sum(uc.values()), perm_yes=uc['perm_yes'],
            perm_no=uc['perm_no'], acks=uc['ack'], guidance_msgs=uc['guidance'],
            message=c['message'], read=c['read'], write=c['write'], run=c['run'], vision=c['vision'],
            denied=c['write_denied'] + c['run_denied'] + c['vision_denied'],
            GalvoScan=mc['GalvoScan'], FindNV=mc['FindNV'], Optimize=mc['Optimize'], ESR=mc['ESR'],
            measurements=sum(mc.values()), first_line=seg[0]['line'], last_line=seg[-1]['line']))

def wcsv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

wcsv(f'{DATA}/sessions.csv', sessions)
wcsv(f'{DATA}/measurements.csv', meas)
wcsv(f'{DATA}/runs.csv', runs)
wcsv(f'{DATA}/failures.csv', failures)
wcsv(f'{DATA}/guidance_raw.csv', guidance)
wcsv(f'{DATA}/hallucinated_turns.csv', hallucinated)
wcsv(f'{DATA}/permissions.csv', perms)

# ---- embeddings: extra lab sittings whose run logs are not in the repo ----
emb = []
for fn in sorted(os.listdir(EMB)):
    t = json.load(open(f'{EMB}/{fn}'))['text']
    v = json.load(open(f'{EMB}/{fn}'))['embeddings']
    saved = [mtype(m.group(1)) for m in SAVED.finditer(t)]
    emb.append(dict(file=fn, chars=len(t), vectors=1 if v and not isinstance(v[0], list) else len(v), dim=len(v) if not isinstance(v[0], list) else len(v[0]),
                    run_ids=sorted(set(re.findall(r'run_\d{8}_\d{6}', t))), windows_paths='\\' in t,
                    hw=bool(HW.search(t)), saved=saved, run_errors=len(re.findall(r'\[System\] Error running command', t))))
logged_runs = {os.path.basename(os.path.dirname(os.path.dirname(f))) for f in log_files()}
extra = [e for e in emb if e['saved'] and not set(e['run_ids']) & logged_runs]

def summ(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return dict(n=len(xs), median=round(st.median(xs), 1), min=round(min(xs), 1), max=round(max(xs), 1),
                mean=round(st.mean(xs), 1))

bytype = defaultdict(list)
for m in meas:
    bytype[m['type']].append(m['script_wall_s'])
bydate = defaultdict(Counter)
for m in meas:
    bydate[m['date']][m['type']] += 1

# consecutive completed-measurement intervals within a sitting (includes LLM + human time)
cyc = []
for s in sessions:
    ts = sorted(m['ts_end'] for m in meas if m['log'] == s['log'] and s['start'] <= m['ts_end'] <= s['end'])
    from datetime import datetime as D
    cyc += [(D.fromisoformat(b) - D.fromisoformat(a)).total_seconds() / 60 for a, b in zip(ts, ts[1:])]

hw_s = [s for s in sessions if s['reached_hardware']]
stats = dict(
    definitions=dict(
        measurement='"[Runner] Saved <type> data to:" line at line start inside logged script stdout',
        reached_hardware='a logged RUN OUTPUT (script stdout) contains "PB connected" (PulseBlaster driver loaded) or "[Runner] Saved"',
        sitting='contiguous part of one log file with no gap > 3 h',
        llm_turn='raw ASSISTANT log entry (not the "(THINK)" echo); one per LLM call',
        guidance='USER message that is neither a permission answer nor a bare acknowledgement (yes/ok/sure/proceed...)'),
    n_log_files=len(log_files()), n_sittings=len(sessions), n_sittings_hw=len(hw_s),
    hw_dates=sorted({s['date'] for s in hw_s}),
    measurements_total=len(meas), measurements_by_type=dict(Counter(m['type'] for m in meas)),
    measurement_dates=len(bydate), measurements_by_date={d: dict(c) for d, c in sorted(bydate.items())},
    script_wall_s_by_type={k: summ(v) for k, v in bytype.items()},
    interval_between_measurements_min=summ(cyc),
    run_commands=len(runs), run_outcomes=dict(Counter(r.get('outcome', 'no output logged') for r in runs)),
    actions_executed=dict(act_exec), actions_proposed_by_llm=dict(act_prop),
    actions_executed_by_version={k: dict(v) for k, v in act_exec_by_ver.items()},
    llm_turns_total=sum(s['llm_turns'] for s in sessions),
    llm_turns_per_sitting=summ([s['llm_turns'] for s in sessions]),
    llm_turns_per_hw_sitting=summ([s['llm_turns'] for s in hw_s]),
    llm_latency_s=summ(llm_latency), human_reply_latency_s=summ(human_latency),
    permission_answers=dict(Counter(p['answer'] for p in perms)),
    permission_answers_by_version={v: dict(Counter(p['answer'] for p in perms if p['version'] == v)) for v in {p['version'] for p in perms}},
    permission_answers_vs_hallucinated_turn={f"{a}|hallucinated={h}": n for (a, h), n in
                                             Counter((p['answer'], p['llm_turn_hallucinated']) for p in perms).items()},
    user_messages=dict(Counter(classify_user(e['text']) for f in log_files() for e in parse(f) if e['role'] == 'USER')),
    vision_executed=act_exec['vision'], vision_denied=act_exec['vision_denied'],
    failures_by_category=dict(Counter(x['category'] for x in failures)),
    failures_by_category_hw=dict(Counter(x['category'] for x in failures if x['hw_session'])),
    hallucinated_llm_turns=len(hallucinated),
    hallucinated_llm_turns_by_version=dict(Counter(h['version'] for h in hallucinated)),
    hallucinated_saved_lines=sum(h['fake_saved_lines'] for h in hallucinated),
    llm_turns_by_version=dict(Counter(s['version'] for s in sessions for _ in range(s['llm_turns']))),
    embeddings=emb, extra_hw_measurements_from_embeddings=[dict(file=e['file'], run_ids=e['run_ids'], saved=e['saved']) for e in extra],
)
json.dump(stats, open(f'{OUT}/stats_logs.json', 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in stats.items() if k not in ('embeddings',)}, indent=1, default=str))
