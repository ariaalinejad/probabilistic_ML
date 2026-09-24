"""Rrs spectra -> the low-dimensional vector the flow conditions on.

Two facts set this design, both measured on the 942 four-WQP matchups:

1. Conditioning on all 63 bands overfits at this sample size -- a ridge probe scores
   CDOM at R^2 = -0.32, worse than predicting the mean. The spectra need compressing.
2. Compressing to *shape alone* is the obvious way to do that and it is wrong here.
   L2-normalising discards the magnitude, and with it TSS (R^2 -0.13) and Secchi
   (-0.03). Restoring log||Rrs|| as one extra feature lifts them to 0.47 and 0.52.

So: 5 EDAA abundances describing the shape, plus one magnitude feature. EDAA rather
than PCA because archetypal abundances live on the simplex and read as water-type
mixing fractions, which is the interpretable story; ablations.py checks that choice
against PCA rather than assuming it.

The endmembers are fit on the training split only and frozen for val/test. Fitting on
everything would leak the test spectra into the representation and quietly flatter
every calibration number downstream.
"""
import numpy as np

from probai_course.probabilistic_ml.NPE.edaa import BlindEDAA

N_COMPONENTS = 5

# EDAA is non-convex, so M restarts are taken and the least-correlated good fit wins.
# Heavier than the class defaults (M=5, T=15) and still ~seconds at n<1000.
EDAA_KWARGS = dict(M=20, T=50, K1=30, K2=30, AA_init=True)


def normalize(X):
    """L2-normalise each spectrum (row). Returns a new float32 array.

    Copied from unmixing_for_owt/spectra_selection.py rather than imported: that module
    pulls in the PACE pipeline's config at import time.
    """
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def split_magnitude(rrs):
    """(n, bands) Rrs -> unit-norm shape (n, bands) and log10 magnitude (n,).

    A handful of GLORIA bands go slightly negative (min ~-5e-5 /sr, sensor noise around
    zero in the blue). Clipped to 0 so the norm stays a sensible scale and the simplex
    fit is not chasing physically impossible reflectance.
    """
    X = np.clip(np.asarray(rrs, dtype=np.float64), 0.0, None)
    norms = np.linalg.norm(X, axis=1)
    # a spectrum of all zeros would divide by zero and has no shape to speak of
    safe = np.where(norms > 0, norms, 1.0)
    return X / safe[:, None], np.log10(safe)


class SpectralFeatures:
    """EDAA endmembers + feature standardisation, fit on train and applied to any split.

    Holds everything the conditioning vector depends on, so a val/test split cannot be
    transformed with anything the training split did not see.
    """

    def __init__(self, p=N_COMPONENTS, seed=0, use_magnitude=True, **edaa_kwargs):
        self.p = p
        self.seed = seed
        self.use_magnitude = use_magnitude
        self.model = BlindEDAA(**{**EDAA_KWARGS, **edaa_kwargs})
        self.endmembers = None      # (bands, p)
        self.mean = self.std = None

    @property
    def n_features(self):
        return self.p + (1 if self.use_magnitude else 0)

    def fit(self, rrs_train):
        """Fit endmembers and the standardiser on the training spectra only."""
        shape, _ = split_magnitude(rrs_train)
        # BlindEDAA wants (bands, pixels) and float64 for the SPAMS init
        E, _ = self.model.solve(Y=np.asfortranarray(shape.T, dtype=np.float64),
                                p=self.p, seed=self.seed)
        self.endmembers = np.asarray(E, dtype=np.float64)

        raw = self._raw(rrs_train)
        self.mean = raw.mean(axis=0)
        # a constant feature would divide by zero; leave it centred at 0 instead
        self.std = np.where(raw.std(axis=0) > 0, raw.std(axis=0), 1.0)
        return self

    def _raw(self, rrs):
        """Unstandardised features: p abundances, then log10 magnitude."""
        if self.endmembers is None:
            raise RuntimeError('fit() must be called before features are computed')
        shape, log_mag = split_magnitude(rrs)
        A = self.model.transform(shape.T, self.endmembers)       # (p, n), on the simplex
        cols = [A.T] + ([log_mag[:, None]] if self.use_magnitude else [])
        return np.concatenate(cols, axis=1).astype(np.float64)

    def transform(self, rrs):
        """(n, bands) Rrs -> (n, n_features) standardised conditioning vectors."""
        return (self._raw(rrs) - self.mean) / self.std

    def fit_transform(self, rrs_train):
        return self.fit(rrs_train).transform(rrs_train)

    def reconstruction_residual(self, rrs):
        """Relative error of the p-endmember fit, per spectrum.

        Spectra the archetypes cannot reconstruct are ones whose conditioning vector is
        a poor summary, so this is the natural flag for "the flow is extrapolating".
        Shape is unit-norm, so the residual norm is already relative.
        """
        shape, _ = split_magnitude(rrs)
        A = self.model.transform(shape.T, self.endmembers)
        return np.linalg.norm(shape.T - self.endmembers @ A, axis=0)


def endmember_max_correlation(E):
    """Largest off-diagonal correlation between endmember spectra.

    Near 1.0 means the fit collapsed onto duplicate archetypes -- the failure mode of a
    bad init, and the reason AA_init is not optional.
    """
    p = E.shape[1]
    return float(np.max(np.corrcoef(E.T) - np.eye(p)))


def selftest(p=N_COMPONENTS, seed=0):
    """Check the EDAA fit is well-formed on the real GLORIA matchups."""
    import probai_course.probabilistic_ml.NPE.gloria as gloria

    rrs, _, _ = gloria.matchups(gloria.WQP_COLUMNS, hydropt_grid=True, drop_gaps=True)
    feat = SpectralFeatures(p=p, seed=seed)
    X = feat.fit_transform(rrs)

    shape, _ = split_magnitude(rrs)
    A = feat.model.transform(shape.T, feat.endmembers)
    resid = feat.reconstruction_residual(rrs)
    corr = endmember_max_correlation(feat.endmembers)

    checks = {
        'abundances sum to 1': np.allclose(A.sum(axis=0), 1.0, atol=1e-4),
        'abundances non-negative': bool((A >= -1e-8).all()),
        'endmembers distinct (corr < 0.99)': corr < 0.99,
        # 0.126 is what random init managed; a real fit should be well under it
        'median residual < 0.126': float(np.median(resid)) < 0.126,
        'features finite': bool(np.isfinite(X).all()),
        'features standardised': np.allclose(X.mean(0), 0, atol=1e-8),
    }

    print(f'{len(rrs)} spectra, {p} endmembers -> {X.shape[1]} features')
    print(f'  endmember max |corr|      {corr:.3f}')
    print(f'  reconstruction residual   median {np.median(resid):.4f}  '
          f'p90 {np.percentile(resid, 90):.4f}')
    for name, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    return all(checks.values())


if __name__ == '__main__':
    raise SystemExit(0 if selftest() else 1)
