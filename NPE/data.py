"""GLORIA matchups, split so the held-out numbers mean something.

The split is grouped by Dataset_ID, not random. ceiling.py established that neighbouring
spectra from the same campaign are spectrally closer than cross-campaign ones -- same
site, same water, same lab. A random split puts near-duplicates on both sides, so the
flow can recognise the campaign rather than invert the optics, and every metric comes
back flattering. Grouping forces generalisation to water the model has not seen.

theta is log10 of the four WQPs throughout. They span 3-5 decades each and are strictly
positive (gloria.matchups(positive_only=True) guarantees it), so the flow works in log
space and results stay comparable with the VAE's log-space metrics.
"""
import numpy as np
from sklearn.model_selection import GroupKFold

import probai_course.probabilistic_ml.NPE.gloria as gloria

WQPS = gloria.WQP_COLUMNS                        # Chla, TSS, aCDOM440, Secchi_depth
GROUP_COL = 'Dataset_ID'
META_COLS = [GROUP_COL, 'Water_type', 'Chl_method']


def load(wqps=WQPS):
    """All matchups with every requested WQP, on hydropt's 400-710 nm / 5 nm grid.

    Returns (rrs (n, 63), theta (n, k) = log10 WQPs, ids (n,), groups (n,), meta).

    The 63-band window is not a preference: on GLORIA's full 350-900 nm grid only 133 of
    the four-WQP rows are free of gaps, against 942 here.
    """
    rrs, labels, ids, meta = gloria.matchups(
        wqps, hydropt_grid=True, drop_gaps=True, extra_cols=META_COLS)
    groups = meta[GROUP_COL].to_numpy()
    return rrs, np.log10(labels), ids, groups, meta


def folds(groups, n_splits=5):
    """Grouped CV index pairs. No Dataset_ID appears in both halves of a fold."""
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(f'{n_groups} campaigns cannot fill {n_splits} grouped folds')
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def split(groups, fold=0, n_splits=5):
    """One (train_idx, test_idx) pair from the grouped CV."""
    return folds(groups, n_splits)[fold]


def holdout(groups, val_frac=0.2, seed=0):
    """Carve a grouped validation set out of a training index set.

    Whole campaigns move together, for the same reason the outer split is grouped.
    Returns (train_idx, val_idx) as positions into `groups`.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)

    # take whole campaigns until the validation share is met
    target = val_frac * len(groups)
    val_groups, taken = set(), 0
    for g in unique:
        if taken >= target:
            break
        val_groups.add(g)
        taken += int((groups == g).sum())

    is_val = np.isin(groups, list(val_groups))
    return np.where(~is_val)[0], np.where(is_val)[0]


def describe():
    """Print what the dataset looks like, and confirm the splits are disjoint."""
    rrs, theta, ids, groups, meta = load()
    print(f'{len(rrs)} matchups, {rrs.shape[1]} bands, '
          f'{len(np.unique(groups))} campaigns')
    print(f'{"WQP":<15}{"min":>8}{"median":>9}{"max":>9}   (log10)')
    for j, col in enumerate(WQPS):
        t = theta[:, j]
        print(f'{col:<15}{t.min():8.2f}{np.median(t):9.2f}{t.max():9.2f}')

    print('\nlog10 label correlations:')
    C = np.corrcoef(theta.T)
    print(' ' * 15 + ''.join(f'{c[:9]:>10}' for c in WQPS))
    for j, col in enumerate(WQPS):
        print(f'{col:<15}' + ''.join(f'{C[j, k]:10.2f}' for k in range(len(WQPS))))
    return rrs, theta, groups


def check_split_disjoint(n_splits=5):
    """No campaign, and no sample, may appear on both sides of any split."""
    _, _, _, groups, _ = load()
    ok = True

    for k, (tr, te) in enumerate(folds(groups, n_splits)):
        shared_rows = set(tr) & set(te)
        shared_groups = set(groups[tr]) & set(groups[te])
        good = not shared_rows and not shared_groups
        ok &= good
        print(f'  [{"PASS" if good else "FAIL"}] fold {k}: '
              f'train {len(tr):4d} / test {len(te):4d}, '
              f'{len(set(groups[te]))} held-out campaigns')

    tr, va = holdout(groups[folds(groups, n_splits)[0][0]])
    inner_good = not set(tr) & set(va)
    ok &= inner_good
    print(f'  [{"PASS" if inner_good else "FAIL"}] inner holdout disjoint '
          f'({len(tr)} train / {len(va)} val)')
    return ok


if __name__ == '__main__':
    describe()
    print()
    raise SystemExit(0 if check_split_disjoint() else 1)
