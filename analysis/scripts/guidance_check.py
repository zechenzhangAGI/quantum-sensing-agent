"""Did the agent follow the operator's parametric guidance?

For every guidance message (see analyze.py) that names coordinates "(x,y)", a frequency range "a GHz to b GHz",
or a microwave power "p dBm", find the next config the agent wrote (WRITE action) before the next guidance message,
and test whether the written config contains those values. Writes data/guidance_checks.csv.
"""
import os, re, csv, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qsa_logs import *

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FREQ = re.compile(r'(\d\.\d+)\s*G?hz\s*(?:to|-)\s*(\d\.\d+)\s*Ghz', re.I)
XY = re.compile(r'\(\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\)')
DBM = re.compile(r'(-?\d+(?:\.\d+)?)\s*dBm', re.I)

def nums(pattern, text):
    return [float(v) for v in re.findall(pattern, text)]

rows = []
for f in log_files():
    E = parse(f)
    for i, e in enumerate(E):
        if e['role'] != 'USER' or classify_user(e['text']) != 'guidance':
            continue
        t = e['text']
        want = {}
        m = FREQ.search(t)
        if m:
            want['freq'] = sorted([float(m.group(1)) * 1e9, float(m.group(2)) * 1e9])
        xy = XY.findall(t)
        if len(xy) == 1:
            want['xy'] = [float(xy[0][0]), float(xy[0][1])]
        p = DBM.search(t)
        if p and 'not above' not in t:
            want['power'] = float(p.group(1))
        if not want:
            continue
        # next written config before the next guidance message
        w = None
        for x in E[i + 1:]:
            if x['role'] == 'USER' and classify_user(x['text']) == 'guidance':
                break
            if x['role'] == 'ACTION' and x['text'].startswith('WRITE file with content: '):
                w = x; break
        got, ok = {}, {}
        if w:
            c = w['text']
            fs, fe = nums(r'"freq_start":\s*(-?[\d.eE+]+)', c), nums(r'"freq_stop":\s*(-?[\d.eE+]+)', c)
            if fs and fe:
                got['freq'] = sorted([fs[0], fe[0]])
            xs = nums(r'"x":\s*(-?[\d.]+)', c); ys = nums(r'"y":\s*(-?[\d.]+)', c)
            if xs and ys:
                got['xy'] = [xs[0], ys[0]]
            pw = nums(r'"power_out":\s*(-?[\d.]+)', c)
            if pw:
                got['power'] = pw[0]
            for k, v in want.items():
                g = got.get(k)
                ok[k] = g is not None and (all(abs(a - b) < 1e-3 * max(1, abs(a)) for a, b in zip(v, g))
                                           if isinstance(v, list) else abs(v - g) < 1e-6)
        rows.append(dict(log=short(f), line=e['line'], ts=str(e['ts']), guidance=t.strip().replace('\n', ' ')[:200],
                         wanted=json.dumps(want), write_line=w['line'] if w else '', written=json.dumps(got),
                         followed=all(ok.values()) if w else 'no write', detail=json.dumps(ok)))

with open(f'{OUT}/data/guidance_checks.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
for r in rows:
    print(r['log'][-30:], r['line'], r['followed'], r['wanted'], '->', r['write_line'], r['written'], '|', r['guidance'][:90])
