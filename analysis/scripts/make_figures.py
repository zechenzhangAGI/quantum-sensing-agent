"""Figures for the QSA technical report (vector PDF + 300 dpi PNG).

Run after analyze.py and esr_fit.py:  .venv/bin/python -I scripts/make_figures.py   (from analysis/)
Inputs: data/sessions.csv, data/measurements.csv, stats_logs.json, esr_fit.json, recovered_data PNG/JSON.
"""
import os, re, csv, json, sys
from datetime import datetime, date
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import esr_fit

A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = f'{A}/figures'
os.makedirs(FIG, exist_ok=True)
from qsa_logs import CASE_DIR

# Validated categorical slots (dataviz reference palette, light mode); text stays in ink colours.
C = {'GalvoScan': '#2a78d6', 'FindNV': '#eb6834', 'Optimize': '#1baf7a', 'ESR': '#eda100'}
ACT = {'message': '#2a78d6', 'read': '#eb6834', 'write': '#1baf7a', 'run': '#eda100', 'vision': '#e87ba4'}
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 300})

def save(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'{FIG}/{name}.png', bbox_inches='tight')
    plt.close(fig)

def arr(s):
    if isinstance(s, list):
        return np.array(s, dtype=float)
    assert '...' not in s
    return np.array([float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)])

def rows(p):
    return list(csv.DictReader(open(p)))

# ---------------------------------------------------------------- Fig 1: timeline
def fig_timeline():
    S, M = rows(f'{A}/data/sessions.csv'), rows(f'{A}/data/measurements.csv')
    st = json.load(open(f'{A}/stats_logs.json'))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.0, 3.5), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1.15], hspace=0.12))
    days = sorted({m['date'] for m in M})
    types = ['GalvoScan', 'FindNV', 'Optimize', 'ESR']
    w = 1.3
    tops = {}
    for d in days:
        x = mdates.date2num(datetime.fromisoformat(d))
        bottom = 0
        for t in types:
            n = sum(1 for m in M if m['date'] == d and m['type'] == t)
            if n:
                ax.bar(x, n, w, bottom=bottom, color=C[t], edgecolor='white', linewidth=0.8)
                bottom += n
        tops[d] = bottom
        ax.text(x, bottom + 0.4 + (1.0 if d == '2025-05-14' else 0), str(bottom), ha='center', va='bottom', fontsize=7, color=INK)
    # galvo scans known only from embedding snapshots (run logs not in the repo)
    for e in st['extra_hw_measurements_from_embeddings']:
        rid = e['run_ids'][0]
        dd = datetime.strptime(rid[4:12], '%Y%m%d')
        x = mdates.date2num(dd)
        ax.bar(x, 1, w, bottom=tops.get(str(dd.date()), 0), color='white', edgecolor=C['GalvoScan'], linewidth=0.9, hatch='////')
    ax.set_ylabel('completed measurements')
    ax.set_ylim(0, 27)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    hs = [plt.Rectangle((0, 0), 1, 1, color=C[t]) for t in types] + \
         [plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor=C['GalvoScan'], hatch='////')]
    ax.legend(hs, ['galvo scan', 'find_nv', 'optimize (z)', 'ESR (CW)', 'galvo scan seen only in an\nembedding snapshot (no run log)'],
              ncol=3, loc='upper left', fontsize=7, handlelength=1.2, columnspacing=1.0)
    # sittings strip: one row per agent version
    vers = ['v0', 'v1', 'v1b', 'v2', 'v3', 'v4']
    vlab = {'v0': 'v0 pre-repo', 'v1': 'v1 Apr refactor', 'v1b': 'v1b 19 Apr', 'v2': 'v2 RAG proto.', 'v3': 'v3 May lab',
            'v4': 'v4 June auto'}
    for s in S:
        v = s['version'].split()[0]
        y = vers.index(v)
        x0 = mdates.date2num(datetime.fromisoformat(s['start']))
        hw = s['reached_hardware'] == 'True'
        ax2.plot(x0, y, marker='o', ms=5.5 if hw else 4.5, mfc=INK2 if hw else 'white', mec=INK2, mew=0.8, ls='none')
    ax2.set_yticks(range(len(vers))); ax2.set_yticklabels([vlab[v] for v in vers], fontsize=7)
    ax2.set_ylim(-0.6, len(vers) - 0.4); ax2.invert_yaxis()
    ax2.xaxis.grid(True, color=GRID, linewidth=0.5); ax2.set_axisbelow(True)
    ax2.plot([], [], 'o', mfc=INK2, mec=INK2, ms=5, label='session reached hardware')
    ax2.plot([], [], 'o', mfc='white', mec=INK2, ms=4.5, label='no hardware output')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.6, 1.02), fontsize=7, ncol=2, handletextpad=0.2)
    ax2.xaxis.set_major_locator(mdates.MonthLocator()); ax2.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.set_xlim(mdates.date2num(date(2025, 2, 14)), mdates.date2num(date(2025, 6, 22)))
    save(fig, 'fig1_timeline')

# ---------------------------------------------------------------- Fig 2: 14 May chain
def fig_chain():
    d = CASE_DIR
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.18)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 0]); e = fig.add_subplot(gs[1, 1])
    for ax, png, title in ((a, 'GalvoScan_plot_20250514_083103.png', '(a) Confocal scan of the field of view'),
                           (b, 'FindNV_plot_20250514_112326.png', '(b) Zoomed scan on candidate 3')):
        ax.imshow(plt.imread(f'{d}/{png}')); ax.axis('off')
        ax.set_title(title, loc='left', color=INK)
    o = json.load(open(f'{d}/optimization_data_20250514_092342.json'))
    fv = arr(o['fluor_vector']); ext = arr(o['extent'])
    z = np.linspace(ext[0], ext[1], len(fv))
    c.plot(z, fv, color=C['Optimize'], lw=2, marker='o', ms=3.5)
    zm = float(o['maximum_point'])
    c.axvline(zm, color=MUTED, lw=0.8, ls='--')
    c.annotate(f'max {float(o["max_fluor"]):.2f} kcps\nat z = {zm:.3f} V', (zm, float(o['max_fluor'])), xytext=(0.76, 8.6), arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6),
               textcoords=('data', 'data'), fontsize=7, color=INK)
    c.set_xlabel('objective z (V)'); c.set_ylabel('fluorescence (kcps)')
    c.set_title('(c) Focus optimization (candidate 1)', loc='left', color=INK)
    c.yaxis.grid(True, color=GRID, lw=0.5)
    res, (f, y, er, p0, p1, p2) = esr_fit.fit()
    ff = np.linspace(f.min(), f.max(), 600)
    e.errorbar(f / 1e3, y, yerr=er, fmt='o', ms=3, color=INK2, ecolor='#b9b8b3', elinewidth=0.7, capsize=0, zorder=3)
    e.plot(ff / 1e3, esr_fit.m2(ff, *p2), color=C['ESR'], lw=2, zorder=2)
    t = res['two_dip_fit']
    for fc in (t['f1_MHz'], t['f2_MHz']):
        e.axvline(fc / 1e3, color=MUTED, lw=0.7, ls=':')
    e.text(0.02, 0.04, f"two-Lorentzian fit\n"
                       f"f$_1$ = {t['f1_MHz']/1e3:.4f}, f$_2$ = {t['f2_MHz']/1e3:.4f} GHz\n"
                       f"Δf = {t['splitting_MHz']:.1f} ± {t['splitting_err_MHz']:.1f} MHz",
           transform=e.transAxes, fontsize=6.3, bbox=dict(fc='white', ec='none', alpha=0.9, pad=1.5), color=INK, va='bottom')
    e.set_xlabel('microwave frequency (GHz)'); e.set_ylabel('normalized fluorescence')
    e.set_title('(d) ODMR on candidate 3', loc='left', color=INK)
    e.yaxis.grid(True, color=GRID, lw=0.5)
    e.set_ylim(0.905, 1.045)
    save(fig, 'fig2_case_chain_14may')

# ---------------------------------------------------------------- Fig 3: actions
def fig_actions():
    st = json.load(open(f'{A}/stats_logs.json'))
    ex, pr = st['actions_executed'], st['actions_proposed_by_llm']
    kinds = ['message', 'read', 'write', 'run', 'vision']
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.55))
    y = np.arange(len(kinds))[::-1]
    for yi, k in zip(y, kinds):
        n, dn = ex.get(k, 0), ex.get(k + '_denied', 0)
        a.barh(yi, n, 0.62, color=ACT[k], edgecolor='white', lw=0.8)
        if dn:
            a.barh(yi, dn, 0.62, left=n, facecolor='white', edgecolor=ACT[k], hatch='////', lw=0.8)
        a.text(n + dn + 4, yi, f'{n}' + (f' + {dn} refused' if dn else ''), va='center', fontsize=7, color=INK)
    a.set_yticks(y); a.set_yticklabels(kinds)
    a.set_xlabel(f"logged actions, all {st['n_log_files']} logs"); a.set_xlim(0, 330)
    a.set_title('(a) actions by type', loc='left', color=INK)
    a.xaxis.grid(True, color=GRID, lw=0.5); a.set_axisbelow(True)
    vers = [('v0 pre-repo', 'v0'), ('v1 Apr refactor', 'v1'), ('v1b 19 Apr', 'v1b'), ('v3 May lab', 'v3')]
    byv = st['actions_executed_by_version']
    for i, (lab, key) in enumerate(vers):
        cnt = next(v for k, v in byv.items() if k.split()[0] == key)
        tot = sum(cnt.get(k, 0) for k in kinds)
        left = 0
        for k in kinds:
            sh = cnt.get(k, 0) / tot
            b.barh(len(vers) - 1 - i, sh, 0.62, left=left, color=ACT[k], edgecolor='white', lw=1.0)
            if sh > 0.07:
                b.text(left + sh / 2, len(vers) - 1 - i, f'{100*sh:.0f}', ha='center', va='center', fontsize=6.5, color=INK)
            left += sh
        b.text(1.02, len(vers) - 1 - i, f'n={tot}', va='center', fontsize=7, color=INK2)
    b.set_yticks(range(len(vers))[::-1]); b.set_yticklabels([v[0] for v in vers])
    b.set_xlim(0, 1); b.set_xticks([0, .25, .5, .75, 1]); b.set_xticklabels(['0', '25', '50', '75', '100%'])
    b.set_title('(b) share of executed actions by agent version (%)', loc='left', color=INK)
    b.legend([plt.Rectangle((0, 0), 1, 1, color=ACT[k]) for k in kinds], kinds, ncol=5, fontsize=7,
             loc='upper center', bbox_to_anchor=(0.45, -0.22), handlelength=1, columnspacing=0.9)
    save(fig, 'fig3_actions')

# ---------------------------------------------------------------- Fig 4: workflow diagram
def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 3.4)); ax.set_xlim(0, 101); ax.set_ylim(0, 45); ax.axis('off')
    W, H = 22, 9
    X = [1, 26, 51, 76]; Y = [31, 17, 3]
    def box(col, row, text, fc='#f3f2ef', ec=MUTED, bold=False, span=1):
        x, y = X[col], Y[row]; w = W * span + 3 * (span - 1)
        ax.add_patch(FancyBboxPatch((x, y), w, H, boxstyle='round,pad=0.3,rounding_size=1.0', fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + H / 2, text, ha='center', va='center', fontsize=6.3, color=INK,
                weight='bold' if bold else 'normal', linespacing=1.25)
    def arrow(pts, color=INK2, ls='-', text=None, tpos=None):
        xs, ys = zip(*pts)
        ax.plot(xs[:-1] + (xs[-1],), ys, color=color, lw=0.8, ls=ls) if len(pts) > 2 else None
        ax.annotate('', pts[-1], pts[-2], arrowprops=dict(arrowstyle='-|>', color=color, lw=0.8, ls=ls, shrinkA=0, shrinkB=0))
        if len(pts) > 2:
            ax.plot(xs[:-1], ys[:-1], color=color, lw=0.8, ls=ls)
        if text:
            ax.text(*tpos, text, fontsize=6, color=color, ha='center')
    box(0, 0, 'Operator\n(chat message)', bold=True)
    box(1, 0, 'Prompt: system rules\n+ full transcript\n(+ retrieved past text, v2 on)')
    box(2, 0, 'Claude (Anthropic API)\nreturns <think> and\n<action> JSON blocks')
    box(3, 0, 'Actions: message, read,\nwrite (config JSON),\nrun, vision')
    box(3, 1, 'Permission prompt yes/no\n(v0-v1b; no gate\nlogged on 14 May)', fc='#fdf3e6', ec=C['ESR'])
    box(2, 1, 'Script wrapper:\ngalvo_scan, find_nv,\noptimize, ESR')
    box(1, 1, 'b26_toolkit / pylabcontrol\nNI DAQ, PulseBlaster,\nR&S MW generator')
    box(0, 1, 'Confocal NV setup\n(room temperature)', bold=True)
    box(1, 2, 'Saved PNG + JSON in the run folder;\nstdout and vision description of the PNG\nre-enter the transcript', span=2)
    box(3, 2, 'Session embedded on exit\n(all-MiniLM-L6-v2)', fc='#eef4fc', ec=C['GalvoScan'])
    m = H / 2
    for c in range(3):
        arrow([(X[c] + W, Y[0] + m), (X[c + 1], Y[0] + m)])
    arrow([(X[3] + W / 2, Y[0]), (X[3] + W / 2, Y[1] + H)])
    for c in (3, 2, 1):
        arrow([(X[c], Y[1] + m), (X[c - 1] + W, Y[1] + m)])
    arrow([(X[0] + W / 2, Y[1]), (X[0] + W / 2, Y[2] + m), (X[1], Y[2] + m)])
    arrow([(X[1] + 2 * W + 3, Y[2] + m), (X[3], Y[2] + m)])
    # feedback: results back into the prompt (gap column between col 1 and col 2)
    arrow([(X[2] - 1.5, Y[2] + H), (X[2] - 1.5, Y[0] - 2.0), (X[1] + W / 2, Y[0] - 2.0), (X[1] + W / 2, Y[0])], ls='--')
    # retrieval in later sessions
    arrow([(X[3] + W, Y[2] + m), (100.3, Y[2] + m), (100.3, 43.5), (X[1] + W / 2, 43.5), (X[1] + W / 2, Y[0] + H)],
          color=C['GalvoScan'], ls=':', text='retrieval of past sessions in later runs (logged only on 2 May, test sessions)',
          tpos=(62, 44.0))
    save(fig, 'fig4_workflow')

# ---------------------------------------------------------------- Fig 5: ESR fit detail
def fig_esr():
    res, (f, y, er, p0, p1, p2) = esr_fit.fit()
    ff = np.linspace(f.min(), f.max(), 600)
    fig, (a, r) = plt.subplots(2, 1, figsize=(3.4, 3.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08))
    a.errorbar(f, y, yerr=er, fmt='o', ms=3, color=INK2, ecolor='#b9b8b3', elinewidth=0.7, zorder=3)
    a.plot(ff, esr_fit.m1(ff, *p1), color=C['GalvoScan'], lw=1.2, ls='--', label='one dip')
    a.plot(ff, esr_fit.m2(ff, *p2), color=C['ESR'], lw=2, label='two dips (shared width)')
    m = res['models']
    a.text(0.02, 0.03, f"χ²/dof: flat {m['flat']['chi2']:.0f}/{m['flat']['dof']}, one {m['one_dip']['chi2']:.0f}/{m['one_dip']['dof']}, "
                       f"two {m['two_dips']['chi2']:.0f}/{m['two_dips']['dof']}\nfit performed for this report",
           transform=a.transAxes, fontsize=6.3, color=INK)
    a.set_ylabel('normalized fluorescence'); a.legend(fontsize=6.5, loc='upper right')
    a.yaxis.grid(True, color=GRID, lw=0.5)
    r.axhline(0, color=MUTED, lw=0.6)
    r.plot(f, (y - esr_fit.m2(f, *p2)) / er, 'o', ms=2.5, color=INK2)
    r.set_ylabel('resid. (σ)'); r.set_xlabel('microwave frequency (MHz)'); r.set_ylim(-3.5, 3.5)
    save(fig, 'fig5_esr_fit')

if __name__ == '__main__':
    fig_timeline(); fig_chain(); fig_actions(); fig_workflow(); fig_esr()
    print(sorted(os.listdir(FIG)))
