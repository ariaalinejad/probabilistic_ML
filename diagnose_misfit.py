"""Stage 1: locate the structural misspecification in the HYDROPT 3-component model.

Background: closure tests show the forward operator recovers its own output exactly and
stays unbiased at 12% noise, yet real GLORIA chl retrievals are 69% off with 39%
collapsing to the bound. So the failure is structural, not noise/optimiser. This script
asks *which* assumption breaks.

1a. Is the RT polynomial being extrapolated? (_validate_bounds is a no-op upstream)
1b. What does the residual look like, and which constant does its shape fingerprint?
"""
import os
import numpy as np

import gloria
import hydropt_flex as hf

PARAMS = ['phyto', 'cdom', 'nap']
BOUND = 1E-6
START = (.01, .01, .01)      # low start: high starts fail even on synthetic data
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.fit_cache.npz')


def _cache_path(rrs):
    """Cache is keyed by row count, so a subset fit cannot clobber the full-set one."""
    base, ext = os.path.splitext(CACHE)
    return f'{base}_{len(rrs)}{ext}'


def fit_all(rrs, wavebands, refit=False, **constants):
    """Invert every spectrum; return best-fit params, model spectra and residuals.

    Cached per row count -- ~1400 inversions is ~15 s and several stages want them.
    """
    path = _cache_path(rrs)
    if not constants and not refit and os.path.exists(path):
        z = np.load(path)
        if z['rrs_shape'].tolist() == list(rrs.shape):
            return z['est'], z['model']

    _, fwd, inv = hf.build(wavebands, **constants)
    est = np.full((len(rrs), 3), np.nan)
    model = np.full(rrs.shape, np.nan)
    for k, y in enumerate(rrs):
        try:
            h = inv.invert(y=y, x=hf.lmfit_params(START), jac=True)
            est[k] = [h.params[p].value for p in PARAMS]
            model[k] = fwd.forward(**dict(zip(PARAMS, est[k])))
        except Exception:
            pass

    # constant scans pass refit=True and never cache; default fits cache per row count
    if not constants and not refit:
        np.savez(path, est=est, model=model, rrs_shape=np.array(rrs.shape))
    return est, model


def iops_for(bio, values):
    """Total (a, bb) per band, water included -- the polynomial's actual input."""
    return bio.sum_iop(**dict(zip(PARAMS, values)))


def main():
    wb = gloria.bands(hydropt_grid=True)
    rrs, wqp, ids = gloria.matchups(['Chla', 'aCDOM440', 'TSS'],
                                    hydropt_grid=True, drop_gaps=True)
    print(f'{len(rrs)} GLORIA matchups on the hydropt grid\n')

    bio, fwd, _ = hf.build(wb)
    fwd.forward(phyto=1., cdom=.1, nap=1.)      # trigger the interpolation cache

    est, model = fit_all(rrs, wb)
    ok = np.isfinite(est).all(axis=1)
    collapsed = est[:, 0] <= BOUND              # chl pinned to the lower bound

    print(f'fits converged: {ok.sum()}/{len(rrs)},  chl collapsed: {collapsed.mean():.1%}')

    # ---- 1a. is the polynomial being extrapolated? -------------------------
    # We do not have the domain the PACE coefficients were fitted on, so use two
    # independent probes: (i) an empirical envelope from the closure set, which the
    # polynomial provably handles, and (ii) a sign check on the RT gradient, which
    # needs no reference domain at all.
    print('\n' + '=' * 72)
    print('1a. polynomial domain check')
    print('=' * 72)

    # (i) empirical envelope from closure-test truths
    rng = np.random.default_rng(1)
    lo, hi = np.percentile(wqp, [2, 98], axis=0)
    truth = 10 ** rng.uniform(np.log10(lo), np.log10(hi), size=(400, 3))
    iop_sim = np.array([iops_for(bio, t) for t in truth])        # (n, 2, bands)
    iop_real = np.array([iops_for(bio, e) for e in est[ok]])

    for j, nm in enumerate(['a', 'bb']):
        s, r = iop_sim[:, j, :], iop_real[:, j, :]
        env_lo, env_hi = s.min(axis=0), s.max(axis=0)
        out = (r < env_lo) | (r > env_hi)
        print(f'  {nm:>3}: real fits outside closure envelope: '
              f'{out.any(axis=1).mean():6.1%} of spectra, {out.mean():5.1%} of band-values')
        print(f'       real range [{r.min():.4g}, {r.max():.4g}]  '
              f'closure range [{s.min():.4g}, {s.max():.4g}]')

    # (ii) physical sign check on the RT gradient: dRrs/da must be < 0, dRrs/dbb > 0.
    # A degree-4 polynomial extrapolated outside its fit domain need not respect this.
    refl = fwd.refl_model
    def sign_violations(iops):
        bad_a = bad_bb = 0
        for x in iops:
            g = refl.gradient(x)                 # (2, bands): d Rrs / d[a, bb]
            bad_a += (g[0] > 0).any()
            bad_bb += (g[1] < 0).any()
        return bad_a / len(iops), bad_bb / len(iops)

    va_r, vb_r = sign_violations(iop_real)
    va_s, vb_s = sign_violations(iop_sim)
    print(f'\n  RT gradient sign violations (dRrs/da>0 or dRrs/dbb<0), any band:')
    print(f'    real best-fit IOPs : dRrs/da>0 {va_r:6.1%}   dRrs/dbb<0 {vb_r:6.1%}')
    print(f'    closure-set IOPs   : dRrs/da>0 {va_s:6.1%}   dRrs/dbb<0 {vb_s:6.1%}')

    # do collapse cases sit disproportionately outside?
    a_real = iop_real[:, 0, :]
    env_hi_a = iop_sim[:, 0, :].max(axis=0)
    outside = ((a_real > env_hi_a) | (a_real < iop_sim[:, 0, :].min(axis=0))).any(axis=1)
    col_ok = collapsed[ok]
    if outside.any():
        print(f'\n  chl-collapse rate inside envelope : {col_ok[~outside].mean():6.1%}')
        print(f'  chl-collapse rate outside envelope: {col_ok[outside].mean():6.1%}')
    else:
        print(f'\n  no real fits fall outside the closure envelope in absorption')

    # ---- 1b. residual structure and attribution ----------------------------
    print('\n' + '=' * 72)
    print('1b. residual structure')
    print('=' * 72)

    resid = rrs - model
    norm = np.linalg.norm(rrs, axis=1, keepdims=True)
    rn = resid / norm                                   # scale-free residual
    good = np.isfinite(rn).all(axis=1)
    rn, est_g = rn[good], est[good]
    print(f'  median ||resid||/||Rrs||: {np.median(np.linalg.norm(rn, axis=1)):.1%}')

    # is the residual systematic, or does it average out?
    mean_r = rn.mean(axis=0)
    print(f'  ||mean residual|| / mean ||residual||: '
          f'{np.linalg.norm(mean_r) / np.linalg.norm(rn, axis=1).mean():.2f}   '
          f'(1.0 = perfectly systematic, ~0 = random)')

    # PCA of the residuals
    u, s, vt = np.linalg.svd(rn - mean_r, full_matrices=False)
    var = s**2 / (s**2).sum()
    print(f'  PCA of centred residuals, variance explained: '
          f'{", ".join(f"{v:.1%}" for v in var[:4])}')

    # ---- which constant does the residual point at? ------------------------
    # At a least-squares optimum the residual is ~orthogonal to d Rrs/d(chl,cdom,nap),
    # so a constant only *explains* residual through the part of its sensitivity that
    # refitting the three state variables cannot already reproduce. Project each
    # sensitivity vector onto the orthogonal complement of the fitted-parameter
    # Jacobian before measuring alignment -- otherwise constants that merely mimic a
    # chl change score spuriously high.
    print('\n  alignment of residual with each constant\'s sensitivity')
    print('  (after removing directions refitting chl/cdom/nap could absorb)\n')

    sub = np.random.default_rng(0).choice(len(rn), min(500, len(rn)), replace=False)
    perturbed = {}
    for name, base in hf.DEFAULTS.items():
        step = .05 * abs(base)
        _, fwd_p, _ = hf.build(wb, **{name: base + step})
        fwd_p.forward(phyto=1., cdom=.1, nap=1.)
        perturbed[name] = (fwd_p, step)

    rows = []
    for name, (fwd_p, step) in perturbed.items():
        cos, frac = [], []
        for k in sub:
            x = dict(zip(PARAMS, est_g[k]))
            if not np.isfinite(list(x.values())).all():
                continue
            base_rrs = fwd.forward(**x)
            d = (fwd_p.forward(**x) - base_rrs) / step          # d Rrs / d theta
            J = fwd.jacobian(**x)                               # (bands, 3)
            q, _ = np.linalg.qr(J)                              # orthonormal basis of span(J)
            d_perp = d - q @ (q.T @ d)                          # component J cannot absorb
            n_perp = np.linalg.norm(d_perp)
            if n_perp < 1e-30:
                continue
            r = rn[k] * norm[good][k]                           # back to absolute Rrs
            r_perp = r - q @ (q.T @ r)
            cos.append(abs(d_perp @ r_perp) / (n_perp * np.linalg.norm(r_perp)))
            frac.append(n_perp / np.linalg.norm(d))             # how much survives projection
        rows.append((name, np.median(cos), np.median(frac)))

    rows.sort(key=lambda t: -t[1])
    print(f'  {"constant":<18}{"|cos| w/ residual":>19}{"indep. of state vars":>23}')
    for name, c, f in rows:
        print(f'  {name:<18}{c:>18.3f}{f:>22.1%}')

    np.savez(os.path.join(os.path.dirname(CACHE), '.stage1_state.npz'),
             est=est, model=model, rrs=rrs, wqp=wqp, ok=ok, collapsed=collapsed)
    return est, model, rrs, wqp


def skill(obs, est):
    """Log-space retrieval metrics; n counts retrievals off the lower bound."""
    m = np.isfinite(est) & (est > BOUND) & np.isfinite(obs) & (obs > 0)
    if m.sum() < 3:
        return dict(n=int(m.sum()), bias=np.nan, mdape=np.nan, rmsle=np.nan)
    o, e = obs[m], est[m]
    lr = np.log10(e / o)
    return dict(n=int(m.sum()), bias=10**np.median(lr),
                mdape=100 * np.median(np.abs(e - o) / o),
                rmsle=float(np.sqrt(np.mean(lr**2))))


def stage1c(n_sub=500):
    """Free one constant at a time -- globally, shared across all spectra -- and see
    how much the misfit drops and whether chl skill improves.

    Fitting a constant per spectrum would be trivially unidentifiable, so the constant
    is scanned on a grid and every spectrum refitted at each value.
    """
    wb = gloria.bands(hydropt_grid=True)
    rrs, wqp, _ = gloria.matchups(['Chla', 'aCDOM440', 'TSS'],
                                  hydropt_grid=True, drop_gaps=True)
    sub = np.random.default_rng(0).choice(len(rrs), min(n_sub, len(rrs)), replace=False)
    rrs, wqp = rrs[sub], wqp[sub]

    grids = {
        'cdom_slope': np.linspace(.010, .025, 7),          # literature range
        'nap_a_slope': np.linspace(.006, .018, 7),
        'nap_bb_exponent': np.linspace(0., 2., 5),
    }

    print('\n' + '=' * 72)
    print(f'1c. free one constant globally (n={len(rrs)} spectra per fit)')
    print('=' * 72)

    baseline = None
    best = {}
    for name, grid in grids.items():
        print(f'\n{name}  (default {hf.DEFAULTS[name]:g})')
        print(f'  {"value":>8}{"med |resid|/|Rrs|":>20}{"chl bias":>11}'
              f'{"chl MdAPE":>11}{"chl n":>8}')
        rows = []
        for v in grid:
            est, model = fit_all(rrs, wb, refit=True, **{name: v})
            rel = np.median(np.linalg.norm(rrs - model, axis=1)
                            / np.linalg.norm(rrs, axis=1))
            sk = skill(wqp[:, 0], est[:, 0])
            rows.append((v, rel, sk))
            flag = '  <- default' if np.isclose(v, hf.DEFAULTS[name], rtol=.02) else ''
            print(f'  {v:>8.4g}{rel:>19.1%}{sk["bias"]:>11.2f}'
                  f'{sk["mdape"]:>10.0f}%{sk["n"]:>8d}{flag}')
        v, rel, sk = min(rows, key=lambda t: t[1])
        best[name] = (v, rel, sk)
        if baseline is None:
            d = fit_all(rrs, wb, refit=True)
            baseline = (np.median(np.linalg.norm(rrs - d[1], axis=1)
                                  / np.linalg.norm(rrs, axis=1)),
                        skill(wqp[:, 0], d[0][:, 0]))

    print('\n' + '-' * 72)
    print(f'baseline (all defaults): resid {baseline[0]:.1%}, '
          f'chl bias {baseline[1]["bias"]:.2f}, MdAPE {baseline[1]["mdape"]:.0f}%, '
          f'n={baseline[1]["n"]}')
    for name, (v, rel, sk) in sorted(best.items(), key=lambda t: t[1][1]):
        print(f'best {name:<16} = {v:<8.4g} resid {rel:.1%}  '
              f'chl bias {sk["bias"]:.2f}, MdAPE {sk["mdape"]:.0f}%, n={sk["n"]}')

    # joint fit of the two slope constants, on the grid neighbourhood of the singles
    print('\njoint scan: cdom_slope x nap_a_slope')
    cg = np.linspace(.010, .025, 6)
    ng = np.linspace(.006, .018, 6)
    out = np.full((len(cg), len(ng)), np.nan)
    best_j = None
    for i, cv in enumerate(cg):
        for j, nv in enumerate(ng):
            est, model = fit_all(rrs, wb, refit=True, cdom_slope=cv, nap_a_slope=nv)
            rel = np.median(np.linalg.norm(rrs - model, axis=1)
                            / np.linalg.norm(rrs, axis=1))
            out[i, j] = rel
            sk = skill(wqp[:, 0], est[:, 0])
            if best_j is None or rel < best_j[0]:
                best_j = (rel, cv, nv, sk)
    rel, cv, nv, sk = best_j
    print(f'  best: cdom_slope={cv:.4g}, nap_a_slope={nv:.4g} -> resid {rel:.1%}, '
          f'chl bias {sk["bias"]:.2f}, MdAPE {sk["mdape"]:.0f}%, n={sk["n"]}')
    return best, best_j


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '1c':
        stage1c()
    else:
        main()
