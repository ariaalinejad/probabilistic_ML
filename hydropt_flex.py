"""hydropt IOP models with their hard-coded constants exposed as arguments.

hydropt's bio_optics functions bury the bio-optical assumptions as literals: the CDOM
spectral slope, the phytoplankton absorption scale and backscatter ratio, the NAP
absorption slope and backscatter shape. Those literals are what we suspect is
misspecified for GLORIA, so they need to be tunable.

Each factory mirrors the upstream `(iop, gradient)` convention that
BioOpticalModel.set_iop expects (hydropt.py:68-78), so these drop straight in via
functools.partial. At default values they reproduce upstream exactly -- see
test_matches_upstream().
"""
import numpy as np

from hydropt.bio_optics import H2O_IOP_DEFAULT, a_phyto_base_HSI

# upstream defaults, from vendor/hydropt-oc/hydropt/bio_optics.py
DEFAULTS = dict(
    cdom_slope=0.017,        # bio_optics.py:72
    phyto_a_scale=0.06,      # bio_optics.py:94
    phyto_bb=0.014 * 0.18,   # bio_optics.py:98
    nap_a_scale=0.041 * 0.75,   # bio_optics.py:57
    nap_a_slope=0.0123,         # bio_optics.py:57
    nap_bb_scale=0.014 * 0.57,  # bio_optics.py:57
    nap_bb_exponent=1.0,        # exponent on (550/wb), implicit 1 upstream
)


def water_flex(*args):
    """Clear natural water. No free constants -- it is a measured table."""
    def iop(*args):
        return H2O_IOP_DEFAULT.T.values

    def gradient(*args):
        return np.full(H2O_IOP_DEFAULT.T.shape, np.nan)

    return iop, gradient


def phyto_flex(*args, a_scale=DEFAULTS['phyto_a_scale'], bb=DEFAULTS['phyto_bb']):
    """Phytoplankton. a_scale multiplies the Ciotti & Cullen (2002) basis vector;
    bb is the (spectrally flat) backscatter per unit chl."""
    base = a_phyto_base_HSI.absorption.values

    def iop(chl=args[0]):
        return np.array([a_scale * chl * base, np.repeat(bb * chl, len(base))])

    def gradient(*args):
        return np.array([a_scale * base, np.repeat(bb, len(base))])

    return iop, gradient


def cdom_flex(*args, wb, slope=DEFAULTS['cdom_slope']):
    """CDOM: exponential absorption with `slope`, no backscatter."""
    def iop(a_440=args[0]):
        return np.array([a_440 * np.exp(-slope * (wb - 440)), np.zeros(len(wb))])

    def gradient(*args):
        d_a = np.exp(-slope * (wb - 440))
        return np.array([d_a, np.zeros(len(d_a))])

    return iop, gradient


def nap_flex(*args, wb, a_scale=DEFAULTS['nap_a_scale'], a_slope=DEFAULTS['nap_a_slope'],
             bb_scale=DEFAULTS['nap_bb_scale'], bb_exponent=DEFAULTS['nap_bb_exponent']):
    """Non-algal particles: exponential absorption, power-law backscatter."""
    def iop(spm=args[0]):
        return spm * np.array([a_scale * np.exp(-a_slope * (wb - 443)),
                               bb_scale * (550 / wb) ** bb_exponent])

    def gradient(*args):
        return np.array([a_scale * np.exp(-a_slope * (wb - 443)),
                         bb_scale * (550 / wb) ** bb_exponent])

    return iop, gradient


# ---------------------------------------------------------------------------

def build(wavebands, components=('phyto', 'cdom', 'nap'), minimizer=None, **constants):
    """A BioOpticalModel + PolynomialForward + InversionModel with the given constants.

    Unknown keys in `constants` raise, so a typo'd constant name fails loudly rather
    than silently leaving the default in place.
    """
    import lmfit
    import hydropt.hydropt as hd
    from functools import partial

    bad = set(constants) - set(DEFAULTS)
    if bad:
        raise TypeError(f'unknown constants: {sorted(bad)}')
    c = {**DEFAULTS, **constants}

    available = {
        'phyto': partial(phyto_flex, a_scale=c['phyto_a_scale'], bb=c['phyto_bb']),
        'cdom': partial(cdom_flex, wb=wavebands, slope=c['cdom_slope']),
        'nap': partial(nap_flex, wb=wavebands, a_scale=c['nap_a_scale'],
                       a_slope=c['nap_a_slope'], bb_scale=c['nap_bb_scale'],
                       bb_exponent=c['nap_bb_exponent']),
    }

    bio = hd.BioOpticalModel()
    bio.set_iop(wavebands=wavebands, water=water_flex,
                **{k: available[k] for k in components})
    fwd = hd.PolynomialForward(bio)
    inv = hd.InversionModel(fwd_model=fwd, minimizer=minimizer or lmfit.minimize)
    return bio, fwd, inv


def lmfit_params(values, names=('phyto', 'cdom', 'nap'), lower=1E-9):
    import lmfit
    p = lmfit.Parameters()
    for k, v in zip(names, values):
        p.add(k, value=v, min=lower)
    return p


def test_matches_upstream(wavebands=None):
    """Defaults must reproduce hydropt's own IOP models exactly."""
    from functools import partial
    from hydropt.bio_optics import phyto as up_phyto, cdom as up_cdom, nap as up_nap

    wb = np.arange(400, 711, 5) if wavebands is None else wavebands
    checks = {
        'phyto': (phyto_flex(None)[0], up_phyto(None)[0], 3.7),
        'cdom': (partial(cdom_flex, wb=wb)(None)[0], partial(up_cdom, wb=wb)(None)[0], .42),
        'nap': (partial(nap_flex, wb=wb)(None)[0], partial(up_nap, wb=wb)(None)[0], 9.1),
    }
    for name, (mine, theirs, val) in checks.items():
        a, b = mine(val), theirs(val)
        if not np.allclose(a, b, rtol=0, atol=0):
            raise AssertionError(f'{name}: max |diff| = {np.abs(a - b).max():.3e}')
    return True


if __name__ == '__main__':
    print('flex IOP models match upstream exactly:', test_matches_upstream())
