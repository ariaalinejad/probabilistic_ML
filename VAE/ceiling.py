"""How good could ANY retrieval method be on GLORIA?

Two spectra that are near-identical in Rrs cannot be told apart by any method that sees
only Rrs. So the spread of their in-situ labels bounds achievable error from below -- with
no forward model involved, which is what makes it a ceiling rather than another statement
about HYDROPT.

Two things have to be controlled for or the number is meaningless:

1. Nearest neighbours are not actually identical. With 1436 spectra in 63-D the k-th
   neighbour can be far away, and its label legitimately differs. So the spread must be
   reported *as a function of neighbour distance* and read at d -> 0, not at whatever
   distance the sample happens to provide. The synthetic control measures exactly this
   bias: with exact labels it should go to ~0 as d -> 0.
2. Same-campaign neighbours are spectrally closer than cross-campaign ones (same site,
   same water). Comparing them at face value would confound "labels agree" with "spectra
   are more similar", so the campaign split is also read at matched distance.
"""
import numpy as np

import  gloria as gloria

WQPS = ['Chla', 'aCDOM440', 'TSS']


def neighbour_pairs(rrs, labels, groups=None, k=12, normalise=False, restrict=None):
    """Per-spectrum k nearest neighbours in Rrs space.

    Returns (rel_dist, dlog, valid): rel_dist is ||xi-xj||/||xi||, dlog is
    |log10(label_j / label_i)| with shape (n, k, n_wqp).

    normalise : compare spectral *shape* only. Magnitude genuinely carries TSS
                information, so raw and shape-only answer different questions.
    restrict  : 'same' / 'diff' -> neighbour must share / not share the group label.
    """
    X = rrs / np.linalg.norm(rrs, axis=1, keepdims=True) if normalise else rrs
    sq = np.einsum('ij,ij->i', X, X)
    d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    np.fill_diagonal(d2, np.inf)
    dist = np.sqrt(np.maximum(d2, 0))

    if restrict is not None:
        if groups is None:
            raise ValueError('restrict= needs groups')
        same = groups[:, None] == groups[None, :]
        dist = np.where(same if restrict == 'same' else ~same, dist, np.inf)

    order = np.argsort(dist, axis=1)[:, :k]
    nd = np.take_along_axis(dist, order, axis=1)
    rel = nd / np.linalg.norm(X, axis=1, keepdims=True)

    lab = np.log10(labels)
    dlog = np.abs(lab[order] - lab[:, None, :])
    valid = np.isfinite(nd) & np.isfinite(dlog).all(axis=2)
    return rel, dlog, valid


def floor_vs_distance(rel, dlog, valid, edges):
    """Median |dlog10 label| in bins of relative spectral distance.

    The lowest-distance bin is the floor estimate; the trend across bins shows how much
    of a larger number is just neighbours not being close enough.
    """
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = valid & (rel >= lo) & (rel < hi)
        if m.sum() < 30:
            rows.append((lo, hi, int(m.sum()), [np.nan] * dlog.shape[2]))
            continue
        rows.append((lo, hi, int(m.sum()),
                     [float(np.median(dlog[:, :, j][m])) for j in range(dlog.shape[2])]))
    return rows


def as_mdape(dlog):
    """Median |log10 ratio| as the equivalent median percentage error."""
    return 100 * (10**dlog - 1)


EDGES = np.array([0., .02, .05, .10, .20, .40, 1.0])


def print_table(rows, title):
    print(f'\n  {title}')
    print(f'    {"rel. distance":<16}{"pairs":>8}' + ''.join(f'{w:>21}' for w in WQPS))
    for lo, hi, n, vals in rows:
        cells = ''.join('           --        ' if not np.isfinite(v)
                        else f'{v:>9.2f} ({as_mdape(v):>5.0f}%)' for v in vals)
        print(f'    {f"{lo:.2f}-{hi:.2f}":<16}{n:>8}{cells}')


def synthetic_control(n=1436, k=12, seed=1):
    """Estimator validation: exact labels, no misspecification, only degeneracy.

    As d -> 0 this must approach ~0. Whatever it shows in the lowest bin is the
    estimator's own bias, and the GLORIA floor has to be read against it.
    """
    import  hydropt_flex as hf

    wb = gloria.bands(hydropt_grid=True)
    _, fwd, _ = hf.build(wb)
    fwd.forward(phyto=1., cdom=.1, nap=1.)

    _, wqp, _ = gloria.matchups(WQPS, hydropt_grid=True, drop_gaps=True)
    lo, hi = np.percentile(wqp, [2, 98], axis=0)
    rng = np.random.default_rng(seed)
    truth = 10 ** rng.uniform(np.log10(lo), np.log10(hi), size=(n, 3))
    rrs = np.array([fwd.forward(phyto=t[0], cdom=t[1], nap=t[2]) for t in truth])

    print('\n' + '=' * 96)
    print('SYNTHETIC CONTROL -- exact labels, no model error, degeneracy only')
    print('=' * 96)
    for norm, tag in [(False, 'raw Rrs'), (True, 'shape only')]:
        rel, dlog, valid = neighbour_pairs(rrs, truth, k=k, normalise=norm)
        print_table(floor_vs_distance(rel, dlog, valid, EDGES), tag)
    return truth, rrs


# A-priori method classes. HPLC separates chlorophyll-a from degradation products and
# accessory pigments and is the reference method; fluorometric and spectrophotometric both
# suffer pheopigment interference. Classified from the method standard, NOT from how well
# hydropt fits -- selecting on residual would manufacture the answer.
CHL_METHOD_CLASS = {
    'CSIRO Chl - HPLC': 'HPLC', 'STIR Chl - HPLC': 'HPLC',
    'U.S. EPA 445.0': 'fluorometric', 'U.S. EPA 447.0': 'fluorometric',
    'UNL Chl - FL': 'fluorometric', 'JAMSTEC Chl - FL': 'fluorometric',
    'NOAA-GLERL Chl - FL': 'fluorometric', 'WFU Chl - FL': 'fluorometric',
    'SCOR-UNESCO': 'spectrophotometric', 'HJ 897-2017': 'spectrophotometric',
    'LabISA-INPE Chl - SP': 'spectrophotometric', 'IUPUI Chl - SP': 'spectrophotometric',
    'UCT Chl - SP': 'spectrophotometric', 'DIN 38412-16:1985-12': 'spectrophotometric',
    'ISO 10260:1992': 'spectrophotometric', 'NEN 6520': 'spectrophotometric',
    'U.S. EPA 446.0': 'spectrophotometric',
    'APHA 10200 H': 'mixed',        # Standard Methods 10200H covers both SP and FL
}


def stage_b(k=12):
    """Skill and floor by model-independent quality strata."""
    from  diagnose_misfit import fit_all, skill

    rrs, wqp, ids, meta = gloria.matchups(
        WQPS, extra_cols=['Chl_method', 'Optical_stability_of_water', 'Dataset_ID'],
        hydropt_grid=True, drop_gaps=True)
    meta = meta.reset_index(drop=True)
    cls = meta['Chl_method'].map(CHL_METHOD_CLASS).fillna('unknown')
    est, _ = fit_all(rrs, gloria.bands(hydropt_grid=True))

    print('\n' + '=' * 96)
    print('STAGE B -- chl skill by a-priori method class (LM inversion)')
    print('=' * 96)
    print(f'  {"class":<20}{"n":>6}{"bias":>8}{"MdAPE":>8}{"RMSLE":>8}{"kNN floor":>12}')
    for c in ['HPLC', 'fluorometric', 'spectrophotometric', 'mixed', 'unknown']:
        m = (cls == c).to_numpy()
        if m.sum() < 40:
            continue
        sk = skill(wqp[m, 0], est[m, 0])
        rel, dlog, valid = neighbour_pairs(rrs[m], wqp[m], k=min(k, m.sum() - 1))
        close = valid & (rel < .10)
        fl = np.median(dlog[:, :, 0][close]) if close.sum() > 30 else np.nan
        fl_s = '   --' if not np.isfinite(fl) else f'{as_mdape(fl):>10.0f}%'
        print(f'  {c:<20}{m.sum():>6}{sk["bias"]:>8.2f}{sk["mdape"]:>7.0f}%'
              f'{sk["rmsle"]:>8.2f}{fl_s}')
    print('  kNN floor = median label disagreement among neighbours within 10% spectral')
    print('  distance, computed inside that stratum only')
    return cls, meta


def stage_c(epochs=300, k=12):
    """Retrain the VAE on the cleanest model-independent subset and read it against
    that subset's own floor."""
    import  vae as vae
    from  diagnose_misfit import fit_all, skill

    cls, meta = stage_b(k=k)
    rrs, wqp, ids, _ = gloria.matchups(
        WQPS, extra_cols=['Dataset_ID'], hydropt_grid=True, drop_gaps=True)

    # clean = reference chl method, and optically stable water. Both are a-priori
    # quality flags from GLORIA itself, independent of how hydropt performs.
    stable = meta['Optical_stability_of_water'].fillna(9).to_numpy() <= 2
    mask = ((cls == 'HPLC') | (cls == 'fluorometric')).to_numpy() & stable
    print(f'\nclean subset: {mask.sum()} / {len(mask)} matchups '
          f'(reference/fluorometric chl + optically stable)')
    if mask.sum() < 200:
        print('  too small to train on; widening to method class only')
        mask = ((cls == 'HPLC') | (cls == 'fluorometric')).to_numpy()
        print(f'  -> {mask.sum()} matchups')

    print('\n' + '=' * 96)
    print('STAGE C -- clean subset vs its own floor')
    print('=' * 96)

    # floor within the clean subset
    rel, dlog, valid = neighbour_pairs(rrs[mask], wqp[mask], k=k)
    close = valid & (rel < .10)
    floor = [np.median(dlog[:, :, j][close]) for j in range(3)]

    est, _ = fit_all(rrs[mask], gloria.bands(hydropt_grid=True))
    model, val, _ = vae.train('constants', epochs=epochs, quiet=True, subset=mask)
    vae_out = vae.evaluate(model, val, label='VAE on clean subset (held-out)')

    print(f'\n  {"WQP":<10}{"kNN floor":>12}{"LM MdAPE":>11}{"VAE MdAPE":>11}'
          f'{"headroom":>11}')
    for j, name in enumerate(WQPS):
        f = as_mdape(floor[j])
        lm = skill(wqp[mask][:, j], est[:, j])['mdape']
        v = vae_out[name]['mdape']
        print(f'  {name:<10}{f:>11.0f}%{lm:>10.0f}%{v:>10.0f}%'
              f'{min(lm, v) - f:>10.0f}%')
    print('\n  headroom = best model MdAPE minus the floor. Near zero means the data,')
    print('  not the model, is the binding constraint on this subset.')


def main(k=12):
    rrs, wqp, ids, meta = gloria.matchups(
        WQPS, extra_cols=['Dataset_ID'], hydropt_grid=True, drop_gaps=True)
    groups = meta['Dataset_ID'].to_numpy()
    print(f'{len(rrs)} matchups, {len(np.unique(groups))} campaigns, k={k}')

    synthetic_control(k=k)

    print('\n' + '=' * 96)
    print('GLORIA -- label spread between spectrally near-identical measurements')
    print('=' * 96)
    for restrict, tag in [(None, 'any neighbour'), ('same', 'same campaign'),
                          ('diff', 'different campaign')]:
        rel, dlog, valid = neighbour_pairs(rrs, wqp, groups, k=k,
                                           normalise=False, restrict=restrict)
        print_table(floor_vs_distance(rel, dlog, valid, EDGES), f'raw Rrs, {tag}')

    print('\nRead the LOWEST distance bin, and subtract the synthetic control at the same')
    print('bin -- that residual is label + matchup error, the part no model can remove.')
    print('Compare against measured skill: chl ~70%, aCDOM440 ~45%, TSS ~48% MdAPE.')


if __name__ == '__main__':
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else 'a'
    {'a': main, 'b': stage_b, 'c': stage_c}[stage]()
