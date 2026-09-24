"""Is the posterior calibrated, and is it as good as the data allows?

Calibration comes first. A retrieval that is 30% off but says so is more useful for
downstream assimilation than one that is 25% off and claims 5%, and calibration is the
thing NPE offers that the deterministic inversion cannot offer at all.

Three questions, in order of how much they distinguish this from the VAE work:

1. Are the credible intervals honest? -- SBC rank ECDF and empirical coverage.
2. Is the posterior wider than the label-noise floor? ceiling.py bounds how well *any*
   method reading only Rrs can do, because near-identical spectra carry genuinely
   different labels. A posterior narrower than that floor is overconfident no matter
   how good its RMSLE looks.
3. Does the joint posterior capture the -0.93 log Secchi / log TSS dependence? A
   diagonal-Gaussian posterior cannot; that is the concrete case for a flow.

Point-estimate skill is reported alongside in the same log-space metrics the VAE used
(VAE/stage3_compare.py), so the numbers are directly comparable.
"""
import argparse
import os

os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np                                                   # noqa: E402

import data                                                          # noqa: E402
import probai_course.probabilistic_ml.NPE.npe as npe                                                           # noqa: E402

WQPS = data.WQPS


# ---------------------------------------------------------------- calibration

def rank_statistics(draws, truth):
    """SBC ranks: where the truth falls among the posterior draws, per WQP.

    Under a calibrated posterior these are uniform on [0, 1]. Systematic deviation is
    the diagnostic: a U shape means intervals are too narrow (overconfident), a hump
    means too wide, a slope means biased.
    """
    return (draws < truth[None, :, :]).mean(axis=0)


def coverage(draws, truth, levels=(0.5, 0.9)):
    """Fraction of truths inside each central credible interval, per WQP.

    Calibrated means the 90% interval contains the truth 90% of the time. Below that is
    overconfidence, which at this sample size is the failure mode to expect.
    """
    out = {}
    for level in levels:
        lo = np.percentile(draws, 100 * (1 - level) / 2, axis=0)
        hi = np.percentile(draws, 100 * (1 + level) / 2, axis=0)
        out[level] = ((truth >= lo) & (truth <= hi)).mean(axis=0)
    return out


def interval_width(draws, level=0.9):
    """Median width of the central credible interval, in decades (theta is log10)."""
    lo = np.percentile(draws, 100 * (1 - level) / 2, axis=0)
    hi = np.percentile(draws, 100 * (1 + level) / 2, axis=0)
    return np.median(hi - lo, axis=0)


# ---------------------------------------------------------------- point skill

def skill(truth, median):
    """Log-space metrics, matching VAE/stage3_compare.py so runs are comparable.

    truth and median are log10 values, so differences are already log ratios.
    """
    lr = median - truth
    return dict(
        bias=float(10 ** np.median(lr)),                       # multiplicative
        mdape=float(100 * np.median(np.abs(10 ** lr - 1))),
        rmsle=float(np.sqrt(np.mean(lr ** 2))),
        r_log=float(np.corrcoef(truth, median)[0, 1]),
    )


# ---------------------------------------------------------------- the floor

def label_noise_floor(rrs, theta, k=8, max_rel_dist=0.05):
    """Spread of labels among spectrally near-identical spectra, in decades.

    The model-free version of ceiling.py's argument: two spectra a retrieval cannot
    tell apart bound how well any Rrs-only method can do. Read as a posterior width --
    a 90% interval much narrower than this is claiming to resolve what the data does
    not distinguish.

    Restricted to neighbours within max_rel_dist so the number reflects near-duplicates
    rather than whatever the k-th neighbour happened to be.
    """
    X = np.asarray(rrs, dtype=np.float64)
    sq = np.einsum('ij,ij->i', X, X)
    d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    np.fill_diagonal(d2, np.inf)
    dist = np.sqrt(np.maximum(d2, 0)) / np.sqrt(sq)[:, None]      # relative distance

    order = np.argsort(dist, axis=1)[:, :k]
    rows = np.arange(len(X))[:, None]
    near = dist[rows, order] <= max_rel_dist

    spread = []
    for j in range(theta.shape[1]):
        d = np.abs(theta[order, j] - theta[:, j][:, None])
        spread.append(float(np.median(d[near])) if near.any() else np.nan)
    return np.array(spread), int(near.sum())


# ---------------------------------------------------------------- reporting

def evaluate_folds(n_splits=5, epochs=200, num_samples=1000, seed=0, **net_kwargs):
    """Train every grouped fold, pool the held-out posteriors, and report.

    Pooling across folds matters here: there are only 19 campaigns and one fold's test
    set is a single campaign, so a per-fold number says as much about which lake was
    held out as about the method.
    """
    rrs, theta, ids, groups, meta = data.load()
    ranks, medians, truths, widths, draws_by_fold = [], [], [], [], []

    for fold in range(n_splits):
        workflow, extractor, test_set, test_idx, _ = npe.train_fold(
            rrs, theta, groups, fold=fold, n_splits=n_splits, epochs=epochs,
            seed=seed, quiet=True, **net_kwargs)

        draws = npe.sample_posterior(workflow, test_set, num_samples=num_samples)
        truth = test_set['theta']

        ranks.append(rank_statistics(draws, truth))
        medians.append(np.median(draws, axis=0))
        truths.append(truth)
        widths.append(interval_width(draws))
        draws_by_fold.append((draws, truth, test_idx))
        print(f'  fold {fold} done ({len(truth)} held-out spectra)')

    return (np.concatenate(ranks), np.concatenate(medians), np.concatenate(truths),
            np.array(widths), draws_by_fold, (rrs, theta))


CACHE = '.posterior_cache.npz'


def cached_posteriors(path=CACHE, n_splits=None, epochs=None, num_samples=None,
                      seed=None, refit=False, **net_kwargs):
    """Held-out posterior draws for every fold, pooled, reused across runs.

    Training all folds takes minutes and the figures are iterated on far more often
    than the model is, so the draws are cached.

    Settings left as None mean "whatever the cache holds" -- so calling this bare from a
    notebook loads the existing run rather than silently retraining because a default
    happened to differ from what was cached. Pass a value explicitly to require it: if
    the cache disagrees, it refits. `refit=True` always retrains.
    """
    requested = dict(n_splits=n_splits, epochs=epochs, num_samples=num_samples,
                     seed=seed)

    if not refit and os.path.exists(path):
        stored = np.load(path, allow_pickle=True)
        mismatch = {k: (int(stored[k]), v) for k, v in requested.items()
                    if v is not None and int(stored[k]) != v}
        if not mismatch:
            return (stored['draws'], stored['truth'], stored['widths'],
                    stored['rrs'], stored['theta'])
        print(f'{path}: refitting, ' + ', '.join(
            f'{k} cached={c} requested={r}' for k, (c, r) in mismatch.items()))

    # defaults apply only once we are actually training
    defaults = dict(n_splits=5, epochs=200, num_samples=1000, seed=0)
    key = {k: (defaults[k] if v is None else v) for k, v in requested.items()}
    n_splits, epochs = key['n_splits'], key['epochs']
    num_samples, seed = key['num_samples'], key['seed']

    _, _, _, widths, by_fold, (rrs, theta) = evaluate_folds(
        n_splits=n_splits, epochs=epochs, num_samples=num_samples, seed=seed,
        **net_kwargs)
    draws = np.concatenate([d for d, _, _ in by_fold], axis=1)
    truth = np.concatenate([t for _, t, _ in by_fold], axis=0)

    np.savez(path, draws=draws, truth=truth, widths=widths, rrs=rrs, theta=theta, **key)
    return draws, truth, widths, rrs, theta


def report(n_splits=5, epochs=200, num_samples=1000, seed=0, refit=False, **net_kwargs):
    all_draws, all_truth, widths, rrs, theta = cached_posteriors(
        n_splits=n_splits, epochs=epochs, num_samples=num_samples, seed=seed,
        refit=refit, **net_kwargs)
    medians = np.median(all_draws, axis=0)
    cov = coverage(all_draws, all_truth)
    floor, n_pairs = label_noise_floor(rrs, theta)

    print(f'\npooled over {n_splits} grouped folds, {len(all_truth)} held-out spectra')

    print('\n--- calibration (the headline) ---')
    print(f'{"WQP":<15}{"cov50":>8}{"cov90":>8}{"width90":>9}{"floor":>8}{"ratio":>8}')
    for j, col in enumerate(WQPS):
        w = float(np.median(widths[:, j]))
        ratio = w / floor[j] if floor[j] > 0 else np.nan
        print(f'{col:<15}{cov[0.5][j]:8.2f}{cov[0.9][j]:8.2f}'
              f'{w:9.2f}{floor[j]:8.2f}{ratio:8.1f}')
    print('  cov50/cov90 should be ~0.50 / ~0.90; below that is overconfident.')
    print(f'  width90 and floor are decades; floor from {n_pairs} near-duplicate pairs.')
    print('  ratio < 1 means the posterior claims to resolve more than the data allows.')

    print('\n--- point-estimate skill (posterior median) ---')
    print(f'{"WQP":<15}{"bias":>8}{"MdAPE":>8}{"RMSLE":>8}{"r_log":>8}')
    for j, col in enumerate(WQPS):
        s = skill(all_truth[:, j], medians[:, j])
        print(f'{col:<15}{s["bias"]:8.2f}{s["mdape"]:8.0f}{s["rmsle"]:8.2f}'
              f'{s["r_log"]:8.2f}')
    print('  NPE always answers: reported fraction 1.00, unlike the LM inversion.')

    print('\n--- posterior dependence structure ---')
    # does the flow reproduce the -0.93 log-Secchi / log-TSS relation?
    j_tss, j_sd = WQPS.index('TSS'), WQPS.index('Secchi_depth')
    within = [np.corrcoef(all_draws[:, i, j_tss], all_draws[:, i, j_sd])[0, 1]
              for i in range(all_draws.shape[1])]
    print(f'  median within-posterior corr(log TSS, log Secchi) = '
          f'{np.nanmedian(within):+.2f}')
    print(f'  marginal corr in the data                         = '
          f'{np.corrcoef(theta[:, j_tss], theta[:, j_sd])[0, 1]:+.2f}')
    print('  a diagonal-Gaussian posterior would be pinned at 0.00 by construction.')

    return dict(coverage=cov, widths=widths, floor=floor, ranks=ranks,
                medians=medians, truths=all_truth, draws=all_draws)


def main():
    # allow_abbrev=False: see the note in npe.main -- a Jupyter kernel's -f would
    # otherwise be taken as an abbreviation of a real flag and fail on its value
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--num-samples', type=int, default=1000)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--refit', action='store_true',
                    help='retrain even if a matching posterior cache exists')
    args, _ = ap.parse_known_args()

    report(n_splits=args.n_splits, epochs=args.epochs, num_samples=args.num_samples,
           seed=args.seed, refit=args.refit, depth=args.depth, width=args.width)


if __name__ == '__main__':
    main()
