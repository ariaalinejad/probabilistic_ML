"""Stage 2: how much of the retrieval error is the model, and how much is the labels?

The 69% chl error treats GLORIA's lab values as truth. GLORIA merges ~25 campaigns using
19 different chl methods, so some of that spread is measurement inconsistency rather than
hydropt being wrong. Stratifying the skill bounds how much.

CONFOUND, stated up front: Chl_method is largely nested within Dataset_ID -- each campaign
uses one method -- so method effect and site/water-type effect cannot be separated here.
Between-dataset spread is a COMBINED upper bound on label + site variability, not an
isolated label-error estimate.
"""
import numpy as np
import pandas as pd

import  gloria as gloria
import  hydropt_flex as hf
from  diagnose_misfit import fit_all, skill, PARAMS, BOUND

STRATA = ['Water_type', 'Water_body_type', 'Optical_stability_of_water',
          'Chl_method', 'aCDOM_method', 'Dataset_ID']
MIN_N = 50


def main():
    wb = gloria.bands(hydropt_grid=True)
    rrs, wqp, ids, meta = gloria.matchups(
        ['Chla', 'aCDOM440', 'TSS'], extra_cols=STRATA,
        hydropt_grid=True, drop_gaps=True)
    meta = meta.reset_index(drop=True)
    print(f'{len(rrs)} matchups\n')

    est, model = fit_all(rrs, wb)
    resid = (np.linalg.norm(rrs - model, axis=1) / np.linalg.norm(rrs, axis=1))

    overall = skill(wqp[:, 0], est[:, 0])
    print(f'overall chl: bias {overall["bias"]:.2f}x, MdAPE {overall["mdape"]:.0f}%, '
          f'RMSLE {overall["rmsle"]:.2f}, n={overall["n"]}/{len(rrs)}, '
          f'median resid {np.median(resid):.1%}\n')

    # nestedness of method within dataset -- quantify the confound rather than assert it
    nest = (meta.groupby('Dataset_ID', observed=True)['Chl_method']
                .nunique(dropna=True))
    print(f'confound check: {(nest <= 1).mean():.0%} of datasets use a single '
          f'Chl_method ({(nest<=1).sum()}/{len(nest)}); '
          f'method and campaign are largely nested\n')

    summary = {}
    for col in STRATA:
        vc = meta[col].value_counts()
        keep = vc[vc >= MIN_N].index
        if len(keep) < 2:
            continue
        print('=' * 78)
        print(f'{col}  ({len(keep)} strata with >= {MIN_N} rows)')
        print(f'  {"stratum":<28}{"n":>6}{"chl bias":>10}{"MdAPE":>8}'
              f'{"RMSLE":>8}{"resid":>8}{"collapse":>10}')
        rows = []
        for v in keep:
            m = (meta[col] == v).to_numpy()
            sk = skill(wqp[m, 0], est[m, 0])
            if sk['n'] < 10:
                continue
            col_rate = (est[m, 0] <= BOUND).mean()
            rows.append((str(v), int(m.sum()), sk, np.median(resid[m]), col_rate))
            print(f'  {str(v)[:27]:<28}{m.sum():>6}{sk["bias"]:>10.2f}'
                  f'{sk["mdape"]:>7.0f}%{sk["rmsle"]:>8.2f}'
                  f'{np.median(resid[m]):>7.1%}{col_rate:>9.0%}')
        if len(rows) >= 2:
            b = np.array([r[2]['bias'] for r in rows])
            a = np.array([r[2]['mdape'] for r in rows])
            summary[col] = (b.max() / b.min(), a.min(), a.max())
            print(f'  -> bias spread {b.max()/b.min():.1f}x across strata, '
                  f'MdAPE {a.min():.0f}-{a.max():.0f}%')
        print()

    print('=' * 78)
    print('between-stratum spread in chl skill\n')
    print(f'  {"grouping":<32}{"bias max/min":>14}{"MdAPE range":>18}')
    for col, (bs, lo, hi) in sorted(summary.items(), key=lambda t: -t[1][0]):
        print(f'  {col:<32}{bs:>13.1f}x{f"{lo:.0f}-{hi:.0f}%":>18}')

    print('\nInterpretation: the spread across Dataset_ID / Chl_method bounds how much of')
    print('the overall error is attributable to label inconsistency + site effects rather')
    print('than the forward model. It cannot separate the two -- see the confound note.')


if __name__ == '__main__':
    main()
