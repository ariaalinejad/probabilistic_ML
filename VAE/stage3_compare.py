"""Stage 3 comparison: physics-decoder VAE variants vs the deterministic hydropt inversion.

Everything is scored on the same held-out split so the comparison is like-for-like. The
deterministic baseline is scored two ways, because its collapse rate makes the naive
number flattering: it only reports skill on spectra whose chl did not hit the lower bound,
while the VAE always returns an estimate.
"""
import numpy as np
import torch

import probai_course.probabilistic_ml.VAE.gloria as gloria
import probai_course.probabilistic_ml.VAE.hydropt_flex as hf
from probai_course.probabilistic_ml.VAE.diagnose_misfit import PARAMS, BOUND, skill
from probai_course.probabilistic_ml.VAE.vae import train, evaluate, load_data, WQPS

VARIANTS = [
    ('none',      'diag'),
    ('constants', 'diag'),
    ('constants', 'lowrank'),
    ('residual',  'lowrank'),
]


def deterministic_baseline(wb, rrs, wqp, **constants):
    _, fwd, inv = hf.build(wb, **constants)
    est = np.full((len(rrs), 3), np.nan)
    for k, y in enumerate(rrs):
        try:
            h = inv.invert(y=y, x=hf.lmfit_params((.01, .01, .01)), jac=True)
            est[k] = [h.params[p].value for p in PARAMS]
        except Exception:
            pass
    return est


def main(epochs=300):
    wb, rrs_tr, wqp_tr, rrs_va, wqp_va = load_data(seed=0)
    print(f'train {len(rrs_tr)} / held-out {len(rrs_va)} spectra\n')

    rows = []

    # ---- deterministic baselines on the same held-out split -----------------
    for label, const in [('hydropt LM (defaults)', {}),
                         ('hydropt LM (Stage 1c slopes)',
                          dict(cdom_slope=.013, nap_a_slope=.0084))]:
        est = deterministic_baseline(wb, rrs_va, wqp_va, **const)
        collapse = (est[:, 0] <= BOUND).mean()
        r = {}
        for j, name in enumerate(WQPS):
            sk = skill(wqp_va[:, j], est[:, j])
            # "all" scores collapsed retrievals as the bound value rather than dropping
            e_all = np.where(np.isfinite(est[:, j]), np.maximum(est[:, j], 1e-4), 1e-4)
            lr = np.log10(e_all / wqp_va[:, j])
            r[name] = dict(bias=sk['bias'], mdape=sk['mdape'], rmsle=sk['rmsle'],
                           rmsle_all=float(np.sqrt(np.mean(lr**2))),
                           frac=sk['n'] / len(rrs_va), cover=np.nan)
        rows.append((f'{label}  [chl collapse {collapse:.0%}]', r))

    # ---- VAE variants -------------------------------------------------------
    for disc, noise in VARIANTS:
        print(f'training VAE: discrepancy={disc}, noise={noise} ...')
        model, val, _ = train(disc, epochs=epochs, noise=noise, quiet=True)
        out = evaluate(model, val, label=f'  [{disc}/{noise}]')
        for name in WQPS:
            out[name]['frac'] = 1.0
            out[name]['rmsle_all'] = out[name]['rmsle']
        rows.append((f'VAE {disc}/{noise}', out))

    # ---- table --------------------------------------------------------------
    print('\n' + '=' * 100)
    print('held-out comparison')
    print('=' * 100)
    for name in WQPS:
        print(f'\n{name}')
        print(f'  {"method":<42}{"bias":>7}{"MdAPE":>8}{"RMSLE":>8}'
              f'{"RMSLE(all)":>12}{"reported":>10}{"90% cov":>9}')
        for label, r in rows:
            d = r[name]
            cov = '   n/a' if not np.isfinite(d['cover']) else f'{d["cover"]:>8.0%}'
            print(f'  {label:<42}{d["bias"]:>7.2f}{d["mdape"]:>7.0f}%{d["rmsle"]:>8.2f}'
                  f'{d["rmsle_all"]:>12.2f}{d["frac"]:>9.0%}{cov}')

    print('\nRMSLE     : only on retrievals the method actually reported')
    print('RMSLE(all): collapsed retrievals scored at the bound -- the honest number for')
    print('            comparing a method that abstains against one that always answers')
    print('reported  : fraction of held-out spectra given a usable (non-collapsed) value')


if __name__ == '__main__':
    main()
