"""Fig. 2 of the report: completed measurements per hardware day, by type (from data/measurements.csv)."""
import csv, collections, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(f'{HERE}/../data/measurements.csv')))
types = [('GalvoScan', 'confocal scan', '#2a78d6'), ('FindNV', 'locate emitter', '#eb6834'),
         ('Optimize', 'optimize focus', '#1baf7a'), ('ESR', 'ODMR sweep', '#eda100')]
days = sorted({r['date'] for r in rows})
count = collections.Counter((r['date'], r['type']) for r in rows)
labels = [f"{int(d[8:])} {['Jan','Feb','Mar','Apr','May','Jun'][int(d[5:7]) - 1]}" for d in days]

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(6.2, 2.1))
x = np.arange(len(days)); bottom = np.zeros(len(days))
for key, name, col in types:
    h = np.array([count[(d, key)] for d in days])
    ax.bar(x, h, bottom=bottom, color=col, width=0.62, label=name, edgecolor='white', linewidth=0.4)
    bottom += h
for xi, tot in zip(x, bottom):
    ax.text(xi, tot + 0.4, f'{int(tot)}', ha='center', va='bottom', fontsize=7, color='#52514e')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('completed measurements'); ax.set_ylim(0, max(bottom) * 1.18)
ax.yaxis.grid(True, color='#e4e3df', lw=0.5); ax.set_axisbelow(True)
ax.legend(ncol=4, frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(0.0, 1.13))
ax.set_xlabel('2025')
fig.tight_layout()
out = f'{HERE}/../figures/fig_deployment'
fig.savefig(out + '.pdf', bbox_inches='tight'); fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
print('total', int(bottom.sum()), 'days', len(days))
