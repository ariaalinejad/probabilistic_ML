"""Differentiable torch port of the HYDROPT forward model.

No surrogate network is needed: PolynomialReflectance.forward (hydropt.py:209-217) is

    Rrs(b) = exp( sum_k C[b,k] * log(a[b])**i_k * log(bb[b])**j_k )

with C a fixed (n_bands, 15) coefficient array interpolated from
PACE_polynom_04_h2o.csv and (i_k, j_k) the degree-4 2D polynomial powers. The IOP models
are closed-form exponentials and power laws. All of it ports directly and stays
differentiable, so the physics can be used as a VAE decoder with exact gradients.

The bio-optical constants are the ones diagnosed in Stage 1 (see hydropt_flex.DEFAULTS);
each can be frozen at hydropt's value or made learnable.
"""
import numpy as np
import torch
import torch.nn as nn

import hydropt_flex as hf

PARAMS = ['phyto', 'cdom', 'nap']


def _reference_model(wavebands):
    """The numpy hydropt model, used to source coefficients and to verify the port."""
    bio, fwd, _ = hf.build(wavebands)
    fwd.forward(phyto=1., cdom=.1, nap=1.)      # triggers interpolation onto wavebands
    return bio, fwd


class HydroptDecoder(nn.Module):
    """Rrs = f(chl, a_cdom440, spm). Input is log10 concentrations, output is Rrs.

    learn: names from hydropt_flex.DEFAULTS to expose as nn.Parameters. Constants are
    stored in log space where they are strictly positive, so they stay positive under
    unconstrained optimisation.
    """

    def __init__(self, wavebands, learn=(), dtype=torch.float64):
        super().__init__()
        self.wavebands = np.asarray(wavebands)
        _, fwd = _reference_model(self.wavebands)

        C = np.asarray(fwd.refl_model._parameters.values, dtype=float)   # (bands, 15)
        powers = np.asarray(fwd.refl_model._powers, dtype=float)         # (15, 2)
        water = np.asarray(hf.water_flex(None)[0](), dtype=float)        # (2, bands)
        phyto_base = np.asarray(hf.a_phyto_base_HSI.absorption.values, dtype=float)

        self.register_buffer('C', torch.tensor(C, dtype=dtype))
        self.register_buffer('pow_a', torch.tensor(powers[:, 0], dtype=dtype))
        self.register_buffer('pow_bb', torch.tensor(powers[:, 1], dtype=dtype))
        self.register_buffer('water', torch.tensor(water, dtype=dtype))
        self.register_buffer('phyto_base', torch.tensor(phyto_base, dtype=dtype))
        self.register_buffer('wb', torch.tensor(self.wavebands, dtype=dtype))

        self.learn = tuple(learn)
        bad = set(self.learn) - set(hf.DEFAULTS)
        if bad:
            raise TypeError(f'unknown constants: {sorted(bad)}')
        for name, value in hf.DEFAULTS.items():
            t = torch.tensor(float(value), dtype=dtype)
            if name == 'nap_bb_exponent':          # can legitimately be 0 -> keep linear
                store, raw = t, True
            else:
                store, raw = torch.log(t), False
            if name in self.learn:
                self.register_parameter(f'_{name}', nn.Parameter(store))
            else:
                self.register_buffer(f'_{name}', store)
            setattr(self, f'_{name}_raw', raw)

    def constant(self, name):
        v = getattr(self, f'_{name}')
        return v if getattr(self, f'_{name}_raw') else torch.exp(v)

    def iops(self, conc):
        """Total (a, bb) including water. conc is (B, 3) in linear units."""
        chl, cdom, spm = conc[:, 0:1], conc[:, 1:2], conc[:, 2:3]

        a = self.constant('phyto_a_scale') * chl * self.phyto_base
        bb = self.constant('phyto_bb') * chl * torch.ones_like(self.phyto_base)

        a = a + cdom * torch.exp(-self.constant('cdom_slope') * (self.wb - 440))

        a = a + spm * self.constant('nap_a_scale') * torch.exp(
            -self.constant('nap_a_slope') * (self.wb - 443))
        bb = bb + spm * self.constant('nap_bb_scale') * (
            550 / self.wb) ** self.constant('nap_bb_exponent')

        return a + self.water[0], bb + self.water[1]

    def forward(self, log_conc):
        """log_conc: (B, 3) natural-log concentrations. Returns Rrs (B, n_bands)."""
        a, bb = self.iops(torch.exp(log_conc))
        la, lbb = torch.log(a), torch.log(bb)
        # features[..., k] = la**i_k * lbb**j_k, contracted with C over k
        feat = la.unsqueeze(-1) ** self.pow_a * lbb.unsqueeze(-1) ** self.pow_bb
        return torch.exp((self.C * feat).sum(-1))


def check_against_numpy(wavebands=None, n=200, seed=0, tol=1e-10):
    """The torch decoder must reproduce hydropt's numpy forward model."""
    wb = hf.np.arange(400, 711, 5) if wavebands is None else np.asarray(wavebands)
    _, fwd = _reference_model(wb)
    dec = HydroptDecoder(wb)

    rng = np.random.default_rng(seed)
    conc = 10 ** rng.uniform(np.log10([.37, .028, .44]), np.log10([143., 5.2, 119.]),
                             size=(n, 3))
    ref = np.array([fwd.forward(**dict(zip(PARAMS, c))) for c in conc])
    got = dec(torch.tensor(np.log(conc), dtype=torch.float64)).detach().numpy()

    rel = np.abs(got - ref) / np.abs(ref)
    return dict(max_rel=float(rel.max()), median_rel=float(np.median(rel)),
                passed=bool(rel.max() < tol))


def check_gradients(wavebands=None, seed=0):
    """Analytic torch gradients must match hydropt's own analytical jacobian."""
    wb = np.arange(400, 711, 5) if wavebands is None else np.asarray(wavebands)
    _, fwd = _reference_model(wb)
    dec = HydroptDecoder(wb)

    rng = np.random.default_rng(seed)
    c = 10 ** rng.uniform(np.log10([.37, .028, .44]), np.log10([143., 5.2, 119.]))
    ref = fwd.jacobian(**dict(zip(PARAMS, c)))            # (bands, 3), d Rrs / d conc

    lc = torch.tensor(np.log(c)[None], dtype=torch.float64, requires_grad=True)
    out = dec(lc)
    got = np.stack([torch.autograd.grad(out[0, i], lc, retain_graph=True)[0][0].numpy()
                    for i in range(out.shape[1])])       # d Rrs / d log conc
    got = got / c                                        # chain rule to d/d conc

    rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)
    return dict(max_rel=float(rel.max()), median_rel=float(np.median(rel)),
                passed=bool(rel.max() < 1e-8))


if __name__ == '__main__':
    f = check_against_numpy()
    print(f'forward  vs numpy: max rel {f["max_rel"]:.2e}, '
          f'median {f["median_rel"]:.2e} -> {"PASS" if f["passed"] else "FAIL"}')
    g = check_gradients()
    print(f'gradient vs hydropt jacobian: max rel {g["max_rel"]:.2e}, '
          f'median {g["median_rel"]:.2e} -> {"PASS" if g["passed"] else "FAIL"}')
