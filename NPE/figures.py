"""Figures for the NPE retrieval: what was predicted, and should it be believed.

Four panels, each answering one question and none repeating another:

  1. retrieval  -- predicted vs in-situ, with 90% credible intervals. The headline.
  2. calibration -- do those intervals mean what they claim? Coverage curve + SBC ranks.
  3. joint      -- the posterior for one spectrum in the TSS/Secchi plane, which is
                   where a flow visibly beats a diagonal-Gaussian posterior.
  4. widths     -- posterior width against the label-noise floor, per WQP.

Design notes, since these are poster-bound: log-log axes with a dashed 1:1 line, error
bars on a random subset only (all 942 would be a grey wash), metrics stated in each
title rather than left to the eye. Colour is used for one thing only -- identity of the
WQP -- and the four hues are assigned in fixed order, never cycled. Grid and axes stay
recessive so the data is the darkest thing on the panel.

Matches VAE/vae.py's plot_retrieval conventions so the two sets of figures can sit side
by side without the reader re-learning the axes.
"""
import argparse
import os

os.environ.setdefault('KERAS_BACKEND', 'torch')

import matplotlib.pyplot as plt                                      # noqa: E402
import matplotlib.ticker as mticker                                  # noqa: E402
import numpy as np                                                   # noqa: E402

import data                                                          # noqa: E402
from probai_course.probabilistic_ml.NPE.evaluate import (cached_posteriors, coverage, label_noise_floor,  # noqa: E402
                      rank_statistics, skill)

WQPS = data.WQPS

# display name + unit per WQP; extends VAE/vae.py's AXIS_LABEL with Secchi
AXIS_LABEL = {
    'Chla': ('chl-a', r'$\mu$g/L'),
    'TSS': ('TSS', 'mg/L'),
    'aCDOM440': (r'$a_{CDOM}(440)$', '1/m'),
    'Secchi_depth': ('Secchi depth', 'm'),
}

# Fixed hue order, assigned by WQP identity and never cycled. blue / red / aqua / yellow,
# not picked by eye: run through the dataviz palette validator over *all* pairs, where it
# passes every check on the light surface. Four mutually CVD-separable hues are hard --
# the obvious blue/orange/green/violet set fails at deltaE 3.2 under protanopia.
#
# Identity never rests on colour alone here anyway: every WQP gets its own subplot with
# its own title and axis labels, which is the secondary encoding the validator asks for.
COLOURS = {'Chla': '#2a78d6', 'TSS': '#e34948',
           'aCDOM440': '#1baf7a', 'Secchi_depth': '#eda100'}

GRID = dict(alpha=.25, lw=.6, color='0.6')


def _decade_ticks(lo, hi, max_ticks=4):
    """A few round tick values spanning [lo, hi], for a log axis narrower than a decade.

    1-2-5 per decade, thinned to at most max_ticks so short ranges get readable labels
    rather than matplotlib's colliding 3x10^0 / 4x10^0 minor labels.
    """
    decades = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
    candidates = np.array([m * 10.0 ** d for d in decades for m in (1, 2, 5)])
    inside = candidates[(candidates >= lo * 0.95) & (candidates <= hi * 1.05)]
    if len(inside) == 0:
        return [float(f'{v:.2g}') for v in np.geomspace(lo, hi, 3)]
    step = max(1, int(np.ceil(len(inside) / max_ticks)))
    return inside[::step]


def _style(ax):
    """Recessive grid and spines, so the marks carry the panel."""
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('0.5')


# ------------------------------------------------------------------ 1. retrieval

def plot_retrieval(draws, truth, n_err=140, seed=0):
    """Posterior median vs in-situ, per WQP, with 90% credible intervals.

    Everything is plotted in linear units on log axes (theta is log10 internally), so
    the reader sees ug/L and mg/L rather than decades.

    Error bars go on a random subset: all 942 would obscure the scatter, and the bars
    are there to show the *scale* of the uncertainty, not to be read one by one.
    """
    med = np.median(draws, axis=0)
    lo, hi = np.percentile(draws, [5, 95], axis=0)
    cov = coverage(draws, truth)

    rng = np.random.default_rng(seed)
    sub = rng.choice(len(truth), min(n_err, len(truth)), replace=False)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (j, name) in zip(axes, enumerate(WQPS)):
        o, e = 10 ** truth[:, j], 10 ** med[:, j]
        ylo, yhi = 10 ** lo[:, j], 10 ** hi[:, j]

        ax.errorbar(o[sub], e[sub],
                    yerr=[e[sub] - ylo[sub], yhi[sub] - e[sub]],
                    fmt='none', ecolor='0.75', elinewidth=.7, alpha=.65, zorder=1)
        ax.scatter(o, e, s=10, alpha=.55, edgecolor='none',
                   color=COLOURS[name], zorder=2)

        lims = [min(o.min(), e.min()) * .8, max(o.max(), e.max()) * 1.25]
        ax.plot(lims, lims, '--', color='0.35', lw=1, zorder=3)

        s = skill(truth[:, j], med[:, j])
        txt, unit = AXIS_LABEL[name]
        ax.set(xscale='log', yscale='log', xlim=lims, ylim=lims,
               xlabel=f'in-situ [{unit}]', ylabel=f'NPE posterior median [{unit}]')
        ax.set_title(f'{txt}\nbias {s["bias"]:.2f}x   RMSLE {s["rmsle"]:.2f}   '
                     f'90% cov {cov[0.9][j]:.0%}', fontsize=10)
        _style(ax)

    fig.suptitle('NPE retrieval vs in-situ, pooled over 5 grouped folds '
                 '(bars: 90% credible interval on a random subset)', y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 2. calibration

def plot_calibration(draws, truth):
    """Two views of whether the intervals are honest.

    Left: coverage against nominal level. The diagonal is perfect calibration; below it
    is overconfidence -- intervals too narrow for how often they miss.

    Right: SBC rank histograms. Uniform means calibrated; a U shape means the truth
    keeps landing in the tails, which is the same overconfidence seen from the side.
    """
    levels = np.linspace(0.05, 0.95, 19)
    ranks = rank_statistics(draws, truth)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], '--', color='0.35', lw=1, label='perfect', zorder=1)
    for j, name in enumerate(WQPS):
        emp = [coverage(draws, truth, levels=(lv,))[lv][j] for lv in levels]
        txt, _ = AXIS_LABEL[name]
        ax.plot(levels, emp, lw=2, color=COLOURS[name], label=txt, zorder=2)
    ax.set(xlabel='nominal credible level', ylabel='empirical coverage',
           xlim=(0, 1), ylim=(0, 1), title='Coverage: below the line is overconfident')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    _style(ax)

    ax = axes[1]
    bins = np.linspace(0, 1, 13)
    for j, name in enumerate(WQPS):
        txt, _ = AXIS_LABEL[name]
        ax.hist(ranks[:, j], bins=bins, histtype='step', lw=1.8, density=True,
                color=COLOURS[name], label=txt)
    ax.axhline(1.0, ls='--', color='0.35', lw=1)
    ax.set(xlabel='posterior rank of the truth', ylabel='density',
           title='SBC ranks: flat is calibrated, U-shaped is too narrow')
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    fig.suptitle('Are the credible intervals honest?', y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- 3. joint

def plot_joint(draws, truth, n_show=3, seed=0):
    """Posterior in the (TSS, Secchi) plane for a few individual spectra.

    The panel that justifies a flow. log Secchi and log TSS correlate at -0.93 in the
    data, so a correct posterior is a tilted ridge. A diagonal-Gaussian posterior -- the
    VAE's mean-field encoder -- is axis-aligned by construction and structurally cannot
    produce this shape, however well it is trained.
    """
    j_tss, j_sd = WQPS.index('TSS'), WQPS.index('Secchi_depth')

    # pick spectra spanning the TSS range, so the panel is not three near-identical clouds
    order = np.argsort(truth[:, j_tss])
    quantiles = np.linspace(0.1, 0.9, n_show)
    picks = order[(quantiles * (len(order) - 1)).astype(int)]

    fig, axes = plt.subplots(1, n_show, figsize=(4.1 * n_show, 4.0))
    for ax, i in zip(np.atleast_1d(axes), picks):
        x, y = 10 ** draws[:, i, j_tss], 10 ** draws[:, i, j_sd]
        r = np.corrcoef(draws[:, i, j_tss], draws[:, i, j_sd])[0, 1]

        ax.scatter(x, y, s=6, alpha=.22, edgecolor='none',
                   color=COLOURS['TSS'], zorder=2, label='posterior draws')
        ax.scatter(10 ** truth[i, j_tss], 10 ** truth[i, j_sd], marker='*', s=190,
                   color='0.15', zorder=4, label='in-situ truth')
        ax.scatter(np.median(x), np.median(y), marker='x', s=70, lw=2,
                   color='#8256A6', zorder=3, label='posterior median')

        ax.set(xscale='log', yscale='log',
               xlabel='TSS [mg/L]', ylabel='Secchi depth [m]',
               title=f'corr(log TSS, log SD) = {r:+.2f}')
        # A posterior spanning under a decade gets matplotlib's log *minor* labels
        # (3x10^0, 4x10^0, ...), which collide into an unreadable smear. Force plain
        # decimal ticks at a few round values inside each axis range instead.
        for axis, lo_v, hi_v in ((ax.xaxis, x.min(), x.max()),
                                 (ax.yaxis, y.min(), y.max())):
            axis.set_major_formatter(mticker.ScalarFormatter())
            axis.set_minor_formatter(mticker.NullFormatter())
            axis.set_major_locator(
                mticker.FixedLocator(_decade_ticks(lo_v, hi_v)))
        _style(ax)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc='best')

    fig.suptitle('Joint posterior per spectrum: the flow captures the TSS/Secchi ridge '
                 'a diagonal Gaussian cannot', y=1.02)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- 4. widths

def plot_widths(draws, rrs, theta):
    """Posterior width against the label-noise floor, per WQP.

    The floor is the median label disagreement between spectrally near-identical
    spectra: what no Rrs-only method can resolve. A bar below its floor would be
    claiming precision the data cannot support. These sit well above it, which says the
    posteriors are wide enough -- the calibration panel is what says whether they are
    wide *enough in the right places*.
    """
    width = np.median(np.percentile(draws, 95, axis=0)
                      - np.percentile(draws, 5, axis=0), axis=0)
    floor, n_pairs = label_noise_floor(rrs, theta)

    x = np.arange(len(WQPS))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(x, width, width=.56, color=[COLOURS[c] for c in WQPS],
           label='90% posterior width', zorder=2)
    ax.bar(x, floor, width=.56, facecolor='none', edgecolor='0.25', lw=1.4,
           hatch='///', label=f'label-noise floor ({n_pairs} near-duplicate pairs)',
           zorder=3)

    for xi, w, f in zip(x, width, floor):
        ax.text(xi, w + .02, f'{w / f:.0f}x', ha='center', fontsize=9, color='0.25')

    ax.set_xticks(x, [AXIS_LABEL[c][0] for c in WQPS])
    ax.set_ylabel('decades (log10)')
    ax.set_title('Posterior width vs what the data can resolve\n'
                 'ratio < 1 would mean claiming more precision than the labels support',
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- driver

FIGURE_DIR = 'figures'


def main():
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--num-samples', type=int, default=500)
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--refit', action='store_true')
    ap.add_argument('--no-save', action='store_true')
    args, _ = ap.parse_known_args()

    draws, truth, widths, rrs, theta = cached_posteriors(
        n_splits=args.n_splits, epochs=args.epochs, num_samples=args.num_samples,
        seed=args.seed, refit=args.refit)
    print(f'{draws.shape[1]} held-out spectra, {draws.shape[0]} posterior draws each')

    figures = {
        '1_retrieval': plot_retrieval(draws, truth, seed=args.seed),
        '2_calibration': plot_calibration(draws, truth),
        '3_joint_posterior': plot_joint(draws, truth, seed=args.seed),
        '4_posterior_width': plot_widths(draws, rrs, theta),
    }

    if not args.no_save:
        os.makedirs(FIGURE_DIR, exist_ok=True)
        for name, fig in figures.items():
            path = os.path.join(FIGURE_DIR, f'{name}.png')
            fig.savefig(path, dpi=160, bbox_inches='tight')
            print(f'  wrote {path}')
    plt.show()


if __name__ == '__main__':
    main()
