"""Two-Lorentzian fit of the 14 May candidate-3 (NV3) ESR spectrum. Fit performed for this report.

Data: recovered_data/run_20250514_082859/data/esr_data_20250514_114852.json (from git commit b316b58).
Model comparison: flat baseline vs one Lorentzian dip vs two Lorentzian dips with a shared width, weighted by
data_err. Uncertainties: covariance (scaled by reduced chi^2 when > 1) and a parametric bootstrap.
Writes esr_fit.json next to stats.json.
"""
import os, re, sys, json
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qsa_logs import CASE_DIR
SRC = f'{CASE_DIR}/esr_data_20250514_114852.json'
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMMA = 2.8  # MHz/G, as specified in the brief

def arr(s):
    assert '...' not in s, 'truncated numpy repr'
    return np.array([float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)])

def load():
    d = json.load(open(SRC))
    return arr(d['frequency']) / 1e6, arr(d['data']), arr(d['data_err'])  # MHz

def lor(f, f0, w):
    return (w / 2) ** 2 / ((f - f0) ** 2 + (w / 2) ** 2)

def m0(f, c):
    return np.full_like(f, c)

def m1(f, c, a, f0, w):
    return c - a * lor(f, f0, w)

def m2(f, c, a1, f1, a2, f2, w):
    return c - a1 * lor(f, f1, w) - a2 * lor(f, f2, w)

def chi2(model, p, f, y, e):
    return float(np.sum(((y - model(f, *p)) / e) ** 2))

def fit():
    f, y, e = load()
    n = len(f)
    res = dict(source=SRC.replace(os.path.expanduser('~'), '~'), n_points=n,
               freq_range_MHz=[float(f.min()), float(f.max())], step_MHz=float(np.diff(f).mean()),
               median_err=float(np.median(e)), label='fit performed for this report')
    p0, _ = curve_fit(m0, f, y, p0=[1.0], sigma=e, absolute_sigma=True)
    p1, _ = curve_fit(m1, f, y, p0=[1.0, 0.03, 2878.0, 4.0], sigma=e, absolute_sigma=True,
                      bounds=([0.9, 0, 2850, 0.5], [1.1, 0.2, 2890, 30]))
    b2 = ([0.9, 0, 2850, 0, 2850, 0.5], [1.1, 0.2, 2890, 0.2, 2890, 30])
    p2, c2 = curve_fit(m2, f, y, p0=[1.005, 0.03, 2863.0, 0.03, 2878.0, 4.0], sigma=e, absolute_sigma=True, bounds=b2)
    out = {}
    for name, model, p in (('flat', m0, p0), ('one_dip', m1, p1), ('two_dips', m2, p2)):
        k = len(p); x2 = chi2(model, p, f, y, e)
        out[name] = dict(params=[float(v) for v in p], k=k, chi2=round(x2, 2), dof=n - k, red_chi2=round(x2 / (n - k), 3),
                         p_value=float(stats.chi2.sf(x2, n - k)), AIC=round(x2 + 2 * k, 2), BIC=round(x2 + k * np.log(n), 2))
    res['models'] = out
    # nested-model F tests
    def ftest(a, b):
        A, B = out[a], out[b]
        F = ((A['chi2'] - B['chi2']) / (B['k'] - A['k'])) / (B['chi2'] / B['dof'])
        return dict(F=round(F, 3), p=float(stats.f.sf(F, B['k'] - A['k'], B['dof'])))
    res['F_flat_vs_two'] = ftest('flat', 'two_dips')
    res['F_one_vs_two'] = ftest('one_dip', 'two_dips')
    # covariance uncertainties, inflated by sqrt(red chi2) if > 1
    s = max(1.0, out['two_dips']['red_chi2']) ** 0.5
    err = np.sqrt(np.diag(c2)) * s
    c, a1, f1, a2, f2, w = p2
    J = np.array([0, 0, -1, 0, 1, 0])  # d(split)/dp
    split_err = float(np.sqrt(J @ c2 @ J) * s)
    split = f2 - f1
    res['two_dip_fit'] = dict(
        baseline=float(c), depth1=float(a1), depth1_err=float(err[1]), f1_MHz=float(f1), f1_err_MHz=float(err[2]),
        depth2=float(a2), depth2_err=float(err[3]), f2_MHz=float(f2), f2_err_MHz=float(err[4]),
        fwhm_MHz=float(w), fwhm_err_MHz=float(err[5]), center_mean_MHz=float((f1 + f2) / 2),
        splitting_MHz=float(split), splitting_err_MHz=split_err,
        B_parallel_G=float(split / (2 * GAMMA)), B_parallel_err_G=float(split_err / (2 * GAMMA)),
        contrast_pct=[float(100 * a1 / c), float(100 * a2 / c)], depth_significance_sigma=[float(a1 / err[1]), float(a2 / err[3])],
        uncertainty_scaling=float(s), gamma_MHz_per_G=GAMMA)
    # parametric bootstrap: resample y ~ N(model, e), refit
    rng = np.random.default_rng(0)
    ymod = m2(f, *p2)
    boots = []
    for _ in range(2000):
        yb = ymod + rng.normal(0, e * s)
        try:
            pb, _ = curve_fit(m2, f, yb, p0=p2, sigma=e, absolute_sigma=True, bounds=b2, maxfev=20000)
            boots.append(pb)
        except RuntimeError:
            pass
    B = np.array(boots)
    sp = B[:, 4] - B[:, 2]
    q = lambda v: [float(x) for x in np.percentile(v, [16, 50, 84])]
    res['bootstrap'] = dict(n=len(B), f1_MHz_16_50_84=q(B[:, 2]), f2_MHz_16_50_84=q(B[:, 4]),
                            splitting_MHz_16_50_84=q(sp), B_parallel_G_16_50_84=q(sp / (2 * GAMMA)),
                            fwhm_MHz_16_50_84=q(B[:, 5]),
                            frac_depth_at_bound=float(np.mean((B[:, 1] < 1e-3) | (B[:, 3] < 1e-3))),
                            frac_width_at_bound=float(np.mean((B[:, 5] < 0.51) | (B[:, 5] > 29.9))))
    # robustness of the splitting to modelling choices
    def m2lin(f, c, s_, a1, f1, a2, f2, w):
        return c + s_ * (f - 2870) - a1 * lor(f, f1, w) - a2 * lor(f, f2, w)
    def m2sep(f, c, a1, f1, w1, a2, f2, w2):
        return c - a1 * lor(f, f1, w1) - a2 * lor(f, f2, w2)
    def m2fix(f, a1, f1, a2, f2, w):
        return 1.0 - a1 * lor(f, f1, w) - a2 * lor(f, f2, w)
    var = {}
    for name, model, p_init, bnd, i1, i2, sl in (
            ('linear_baseline', m2lin, [1.005, 0, 0.03, 2863, 0.03, 2878, 4], ([0.9, -1, 0, 2850, 0, 2850, 0.5], [1.1, 1, 0.2, 2890, 0.2, 2890, 30]), 3, 5, slice(None)),
            ('separate_widths', m2sep, [1.005, 0.03, 2863, 4, 0.03, 2878, 4], ([0.9, 0, 2850, 0.5, 0, 2850, 0.5], [1.1, 0.2, 2890, 30, 0.2, 2890, 30]), 2, 5, slice(None)),
            ('baseline_fixed_1', m2fix, [0.03, 2863, 0.03, 2878, 4], ([0, 2850, 0, 2850, 0.5], [0.2, 2890, 0.2, 2890, 30]), 1, 3, slice(None)),
            ('drop_2852_2858MHz', m2, list(p2), b2, 2, 4, (f < 2851.5) | (f > 2858.5))):
        pp, cc = curve_fit(model, f[sl], y[sl], p0=p_init, sigma=e[sl], absolute_sigma=True, bounds=bnd, maxfev=20000)
        J = np.zeros(len(pp)); J[i1], J[i2] = -1, 1
        var[name] = dict(f1_MHz=round(float(pp[i1]), 2), f2_MHz=round(float(pp[i2]), 2),
                         splitting_MHz=round(float(pp[i2] - pp[i1]), 2), splitting_err_MHz=round(float(np.sqrt(J @ cc @ J)), 2),
                         chi2=round(chi2(model, pp, f[sl], y[sl], e[sl]), 2), n=int(np.sum(np.ones_like(f)[sl])))
    res['robustness'] = var
    return res, (f, y, e, p0, p1, p2)

if __name__ == '__main__':
    res, _ = fit()
    json.dump(res, open(f'{OUT}/esr_fit.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))
