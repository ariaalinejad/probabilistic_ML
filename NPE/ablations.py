"""Does each design choice earn its place?

Four questions a reader will ask about the conditioning vector, each answered by
re-running the same grouped CV with one thing changed:

1. magnitude -- 5 abundances + log||Rrs|| against abundances alone. A ridge probe said
   dropping magnitude costs TSS (R^2 0.47 -> -0.13) and Secchi (0.52 -> -0.03). If the
   flow recovers those from shape alone, the extra feature is unnecessary.
2. EDAA vs PCA at the same dimension. Archetypal abundances are interpretable as water
   type mixing fractions, but interpretability is not free if it costs accuracy.
3. raw 63 bands. Expected to overfit at ~750 training spectra -- that is the argument
   for compressing at all, and it should be measured rather than asserted.
4. 3 WQPs vs 4. Dropping Secchi takes the matchup count from 942 to 1436. More data for
   three parameters, against four parameters from less -- worth knowing which wins.

Everything is scored on the same grouped folds, so the comparison is like-for-like.
"""
import argparse
import os

os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np                                                   # noqa: E402

import data                                                          # noqa: E402
import probabilistic_ml.NPE.npe as npe                                                           # noqa: E402
from probabilistic_ml.NPE.evaluate import coverage, interval_width, skill                 # noqa: E402


class PCAFeatures:
    """Same interface as SpectralFeatures, but linear PCA on the unit-norm spectra.

    The control for EDAA: identical dimension, identical magnitude feature, no simplex
    constraint and no interpretability.
    """

    def __init__(self, p=5, seed=0, use_magnitude=True):
        self.p, self.use_magnitude = p, use_magnitude
        self.components = self.centre = None
        self.mean = self.std = None

    def fit(self, rrs):
        from probabilistic_ml.NPE.features import split_magnitude
        shape, _ = split_magnitude(rrs)
        self.centre = shape.mean(axis=0)
        _, _, Vt = np.linalg.svd(shape - self.centre, full_matrices=False)
        self.components = Vt[:self.p]
        raw = self._raw(rrs)
        self.mean, std = raw.mean(axis=0), raw.std(axis=0)
        self.std = np.where(std > 0, std, 1.0)
        return self

    def _raw(self, rrs):
        from probabilistic_ml.NPE.features import split_magnitude
        shape, log_mag = split_magnitude(rrs)
        scores = (shape - self.centre) @ self.components.T
        cols = [scores] + ([log_mag[:, None]] if self.use_magnitude else [])
        return np.concatenate(cols, axis=1)

    def transform(self, rrs):
        return (self._raw(rrs) - self.mean) / self.std


class RawFeatures:
    """All 63 bands, standardised. The no-compression control."""

    def __init__(self, **_):
        self.mean = self.std = None

    def fit(self, rrs):
        X = np.asarray(rrs, dtype=np.float64)
        self.mean, std = X.mean(axis=0), X.std(axis=0)
        self.std = np.where(std > 0, std, 1.0)
        return self

    def transform(self, rrs):
        return (np.asarray(rrs, dtype=np.float64) - self.mean) / self.std


def run_variant(rrs, theta, groups, extractor_factory, n_splits=5, epochs=200,
                num_samples=500, seed=0, **net_kwargs):
    """Grouped CV for one feature definition; returns pooled draws and truths."""
    draws, truths, widths = [], [], []

    for fold in range(n_splits):
        train_idx, test_idx = data.split(groups, fold=fold, n_splits=n_splits)
        inner_tr, inner_va = data.holdout(groups[train_idx], seed=seed)
        fit_idx, val_idx = train_idx[inner_tr], train_idx[inner_va]

        import keras
        keras.utils.set_random_seed(seed)

        ex = extractor_factory().fit(rrs[fit_idx])
        train_set = npe.as_dataset(theta[fit_idx], ex.transform(rrs[fit_idx]))
        val_set = npe.as_dataset(theta[val_idx], ex.transform(rrs[val_idx]))
        test_set = npe.as_dataset(theta[test_idx], ex.transform(rrs[test_idx]))

        wf = npe.build_workflow(**net_kwargs)
        wf.fit_offline(data=train_set, epochs=epochs, batch_size=64,
                       validation_data=val_set, verbose=0)

        d = npe.sample_posterior(wf, test_set, num_samples=num_samples)
        draws.append(d)
        truths.append(test_set['theta'])
        widths.append(interval_width(d))

    return (np.concatenate(draws, axis=1), np.concatenate(truths, axis=0),
            np.array(widths))


def compare(n_splits=5, epochs=200, num_samples=500, seed=0, **net_kwargs):
    from probabilistic_ml.NPE.features import SpectralFeatures

    rrs, theta, ids, groups, meta = data.load()

    variants = {
        'EDAA-5 + magnitude': lambda: SpectralFeatures(p=5, seed=seed),
        'EDAA-5 shape only': lambda: SpectralFeatures(p=5, seed=seed,
                                                      use_magnitude=False),
        'PCA-5 + magnitude': lambda: PCAFeatures(p=5, seed=seed),
        'raw 63 bands': lambda: RawFeatures(),
    }

    print(f'{len(rrs)} matchups, {n_splits} grouped folds, {epochs} epochs each\n')
    results = {}
    for name, factory in variants.items():
        d, t, w = run_variant(rrs, theta, groups, factory, n_splits=n_splits,
                              epochs=epochs, num_samples=num_samples, seed=seed,
                              **net_kwargs)
        med = np.median(d, axis=0)
        cov = coverage(d, t)
        results[name] = dict(
            rmsle=[skill(t[:, j], med[:, j])['rmsle'] for j in range(t.shape[1])],
            r_log=[skill(t[:, j], med[:, j])['r_log'] for j in range(t.shape[1])],
            cov90=cov[0.9], width90=np.median(w, axis=0))
        print(f'  {name} done')

    for metric, label, fmt in [('rmsle', 'RMSLE (decades, lower better)', '8.2f'),
                               ('r_log', 'r_log (higher better)', '8.2f'),
                               ('cov90', '90% coverage (target 0.90)', '8.2f')]:
        print(f'\n--- {label} ---')
        print(f'{"variant":<22}' + ''.join(f'{c[:9]:>10}' for c in data.WQPS))
        for name, r in results.items():
            print(f'{name:<22}' + ''.join(f'{v:10.2f}' for v in r[metric]))

    return results


def wqp_count_comparison(epochs=200, seed=0, n_splits=5):
    """Four WQPs from 942 matchups against three from 1436.

    Secchi is the sparsest label; asking for it costs a third of the data. This is the
    only ablation that changes the dataset rather than the features.
    """
    from probabilistic_ml.NPE.features import SpectralFeatures

    print('\n--- WQP set (changes the dataset size) ---')
    for wqps in (data.WQPS, ['Chla', 'TSS', 'aCDOM440']):
        rrs, theta, ids, groups, meta = data.load(wqps=wqps)
        d, t, _ = run_variant(rrs, theta, groups,
                              lambda: SpectralFeatures(p=5, seed=seed),
                              n_splits=n_splits, epochs=epochs, seed=seed)
        med = np.median(d, axis=0)
        label = f'{len(wqps)} WQPs (n={len(rrs)})'
        print(f'{label:<22}' + ''.join(
            f'{c[:6]}={skill(t[:, j], med[:, j])["rmsle"]:.2f}  '
            for j, c in enumerate(wqps)))


def main():
    # allow_abbrev=False: see the note in npe.main -- a Jupyter kernel's -f would
    # otherwise be taken as an abbreviation of a real flag and fail on its value
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--num-samples', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--wqp-counts', action='store_true',
                    help='also compare 4-WQP against 3-WQP datasets')
    args, _ = ap.parse_known_args()

    compare(n_splits=args.n_splits, epochs=args.epochs,
            num_samples=args.num_samples, seed=args.seed)
    if args.wqp_counts:
        wqp_count_comparison(epochs=args.epochs, seed=args.seed,
                             n_splits=args.n_splits)


if __name__ == '__main__':
    main()
