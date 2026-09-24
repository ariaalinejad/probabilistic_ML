"""GLORIA loading and matchup selection, shared by the diagnostics and the VAE.

Extracted from read_gloria.py so the analyses don't each re-read ~40 MB of CSV and
re-run its plotting cells.
"""
import os
from functools import lru_cache

import numpy as np
import pandas as pd

GLORIA_DIR = '/home/ariaa/smallSatLab/data/GLORIA_2022'

# the four WQPs of interest. Units: ug/L, mg/L, 1/m, m
WQP_COLUMNS = ['Chla', 'TSS', 'aCDOM440', 'Secchi_depth']

# metadata worth carrying through for stratified skill analysis
META_COLUMNS = ['Water_body_type', 'Water_type', 'Optical_stability_of_water',
                'Measurement_method', 'Chl_method', 'TSS_method', 'aCDOM_method',
                'Dataset_ID', 'Country']

# hydropt's HSI grid: the PACE polynomial is defined on 400-710 nm at 5 nm
HYDROPT_BANDS = np.arange(400, 711, 5)


@lru_cache(maxsize=1)
def load():
    """Merged Rrs + WQP + metadata + QC table, and the Rrs waveband axis.

    Returns (dataset, clean, rrs_bands). `clean` is the QC-passing subset.
    Cached: the CSVs are large and every caller wants the same table.
    """
    rrs_data = pd.read_csv(os.path.join(GLORIA_DIR, 'GLORIA_Rrs.csv'))
    # 350-900 nm at 1 nm steps; column 0 is GLORIA_ID
    rrs_bands = (rrs_data.columns[1:].str.removeprefix('Rrs_').astype(int).to_numpy())

    meta = pd.read_csv(os.path.join(GLORIA_DIR, 'GLORIA_meta_and_lab.csv'), low_memory=False)
    keep = ['GLORIA_ID'] + WQP_COLUMNS + META_COLUMNS

    # one row per spectrum, but the file ends with a stray all-NaN row
    qc_flags = pd.read_csv(os.path.join(GLORIA_DIR, 'GLORIA_qc_flags.csv')).dropna(subset='GLORIA_ID')

    # merge on ID rather than position: the QC table is not row-aligned with the others
    dataset = (rrs_data
               .merge(meta[keep], on='GLORIA_ID')
               .merge(qc_flags[['GLORIA_ID', 'Flagged']], on='GLORIA_ID'))

    return dataset, dataset[dataset['Flagged'] == 0], rrs_bands


def band_index(rrs_bands, wavebands=HYDROPT_BANDS):
    """Positions of `wavebands` within `rrs_bands`.

    GLORIA is 1 nm-gridded over 350-900, so hydropt's 5 nm grid is a strict subset --
    subset rather than interpolate.
    """
    idx = np.searchsorted(rrs_bands, wavebands)
    if not np.array_equal(rrs_bands[idx], wavebands):
        raise ValueError('wavebands are not a subset of rrs_bands')
    return idx


def matchups(wqps=WQP_COLUMNS, qc=True, positive_only=True, extra_cols=None,
             hydropt_grid=False, drop_gaps=False):
    """Spectra + labels for the requested WQPs, dropping rows missing any of them.

    Returns (rrs, labels, ids) or (rrs, labels, ids, meta) when extra_cols is given.
    `rrs` is (n, n_bands), `labels` is (n, len(wqps)), `meta` is a DataFrame.

    Asking for all four WQPs at once costs a lot of rows -- Secchi is the sparsest.

    A handful of rows report exactly 0 for a WQP (below detection limit, most
    likely). positive_only drops them, since these get log-transformed downstream.

    hydropt_grid subsets the spectra to 400-710 nm at 5 nm; drop_gaps additionally
    removes spectra with non-finite values left in that window.
    """
    dataset, clean, rrs_bands = load()
    wqps = list(wqps)

    df = (clean if qc else dataset).dropna(subset=wqps)
    if positive_only:
        df = df[(df[wqps] > 0).all(axis=1)]

    rrs = df.filter(like='Rrs_').to_numpy()
    if hydropt_grid:
        rrs = rrs[:, band_index(rrs_bands)]

    labels, ids = df[wqps].to_numpy(), df['GLORIA_ID'].to_numpy()
    meta = df[list(extra_cols)] if extra_cols is not None else None

    if drop_gaps:
        ok = np.isfinite(rrs).all(axis=1)
        rrs, labels, ids = rrs[ok], labels[ok], ids[ok]
        if meta is not None:
            meta = meta[ok]

    return (rrs, labels, ids) if meta is None else (rrs, labels, ids, meta)


def bands(hydropt_grid=False):
    """The waveband axis, full GLORIA grid or hydropt's subset."""
    return HYDROPT_BANDS.copy() if hydropt_grid else load()[2]
