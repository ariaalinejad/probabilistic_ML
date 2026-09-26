"""Physics-decoder VAE for water-quality retrieval from GLORIA Rrs.

q(z | Rrs) is an MLP encoder over z = standardised log (chl, a_cdom440, spm); the decoder
is HYDROPT itself (physics_decoder.HydroptDecoder), so the latent space is the physical
state and the posterior is directly interpretable as a retrieval with uncertainty.

Training is unsupervised -- only Rrs is used. In-situ WQPs are held out entirely and used
only to evaluate, which is the point: the physics supplies the inductive bias that
supervised regression would have to learn from scarce matchups.

Discrepancy modes, chosen from the Stage 1 diagnosis:
  none       pure physics, hydropt's constants frozen
  constants  cdom_slope and nap_a_slope learned globally -- Stage 1b showed these are the
             residual directions refitting the state variables cannot absorb, and Stage 1a
             ruled out the RT polynomial
  residual   'constants' plus a low-rank additive correction on log Rrs, regularised
             toward zero so the physics stays dominant
"""

# %% Imports
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import gloria as gloria
from physics_decoder import HydroptDecoder

# %% constants
WQPS = ['Chla', 'aCDOM440', 'TSS']

fig_folder = "/home/ariaa/smallSatLab/output_figures/physics_vae"

save_figures = True

DTYPE = torch.float64          # the degree-4 log-polynomial has little headroom in f32

# generous physical box on log concentration. The polynomial overflows well outside any
# real water (chl ~1e4 ug/L gives Rrs ~1e36), so clamp before it can poison gradients.
LOG_CONC_MIN, LOG_CONC_MAX = np.log(1e-4), np.log(1e4)

AXIS_LABEL = {'Chla': ('chl-a', r'$\mu$g/L'),
              'aCDOM440': (r'$a_{CDOM}(440)$', '1/m'),
              'TSS': ('TSS', 'mg/L')}

# %% functions
class Encoder(nn.Module):
    def __init__(self, n_bands, n_latent=3, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bands, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 2 * n_latent))
        self.n_latent = n_latent

    def forward(self, x):
        h = self.net(x)
        mu, log_sd = h[:, :self.n_latent], h[:, self.n_latent:]
        return mu, log_sd.clamp(-6., 2.)

class PhysicsVAE(nn.Module):
    def __init__(self, wavebands, z_loc, z_scale, discrepancy='constants', rank=2,
                 noise='lowrank', noise_rank=4, resid_modes=None):
        super().__init__()
        n_bands = len(wavebands)
        learn = () if discrepancy == 'none' else ('cdom_slope', 'nap_a_slope')
        self.encoder = Encoder(n_bands).to(DTYPE)
        self.decoder = HydroptDecoder(wavebands, learn=learn, dtype=DTYPE)
        self.discrepancy = discrepancy
        self.noise = noise

        if noise == 'lowrank':
            if resid_modes is not None:
                U0 = torch.tensor(resid_modes[:noise_rank].T, dtype=DTYPE)
            else:
                U0 = torch.randn(n_bands, noise_rank, dtype=DTYPE) * .01
            self.U = nn.Parameter(U0.contiguous())

        # standardised latent -> natural-log concentration
        self.register_buffer('z_loc', torch.tensor(z_loc, dtype=DTYPE))
        self.register_buffer('z_scale', torch.tensor(z_scale, dtype=DTYPE))

        # observation noise: per-band, relative to each spectrum's own mean Rrs
        self.log_sigma_rel = nn.Parameter(torch.full((n_bands,), np.log(.05),
                                                     dtype=DTYPE))
        if discrepancy == 'residual':
            self.res_basis = nn.Parameter(torch.zeros(rank, n_bands, dtype=DTYPE))
            self.res_head = nn.Linear(3, rank).to(DTYPE)
            nn.init.zeros_(self.res_head.weight)
            nn.init.zeros_(self.res_head.bias)

    def decode(self, z):
        log_conc = (self.z_loc + self.z_scale * z).clamp(LOG_CONC_MIN, LOG_CONC_MAX)
        rrs = self.decoder(log_conc)
        if self.discrepancy == 'residual':
            # additive in log space, low rank, driven by the state itself
            rrs = rrs * torch.exp(self.res_head(z) @ self.res_basis)
        return rrs, log_conc

    def _nll(self, rrs, rrs_hat):
        """Negative log-likelihood of the residual.

        diag: independent per-band noise. This is the wrong model here -- Stage 1b showed
        the residual is systematic (77% of its variance in two PCA modes), and treating
        structured model error as 63 independent observations makes the posterior far too
        tight.

        lowrank: Sigma = diag(sigma^2) + U U^T, with U initialised from the residual PCA
        modes. Correlated error along those modes counts as ~one observation rather than
        63, so the posterior widens in exactly the directions the model is wrong.
        """
        s = rrs.mean(dim=1, keepdim=True)                 # per-spectrum scale
        r = (rrs - rrs_hat) / s                           # scale-free residual
        d = self.log_sigma_rel.exp() ** 2

        if self.noise == 'diag':
            return (.5 * r**2 / d + .5 * torch.log(d)).sum(1) + rrs.shape[1] * s.log()[:, 0]

        M = torch.diag(d) + self.U @ self.U.T
        L = torch.linalg.cholesky(M)
        sol = torch.cholesky_solve(r.unsqueeze(-1), L).squeeze(-1)
        quad = (r * sol).sum(1)
        logdet = 2 * torch.log(torch.diagonal(L)).sum()
        return .5 * quad + .5 * logdet + rrs.shape[1] * s.log()[:, 0]

    def forward(self, x_std, rrs):
        mu, log_sd = self.encoder(x_std)
        z = mu + torch.randn_like(mu) * log_sd.exp()
        rrs_hat, _ = self.decode(z)
        nll = self._nll(rrs, rrs_hat)
        kl = (.5 * (mu**2 + (2 * log_sd).exp() - 1) - log_sd).sum(1)
        return nll, kl, mu, log_sd


def load_data(seed=0, val_frac=.25, subset=None):
    """subset: optional boolean mask over the matchups, applied before splitting, so a
    restricted-quality subset gets its own train/val split rather than a filtered val."""
    wb = gloria.bands(hydropt_grid=True)
    rrs, wqp, ids = gloria.matchups(WQPS, hydropt_grid=True, drop_gaps=True)
    if subset is not None:
        subset = np.asarray(subset, dtype=bool)
        rrs, wqp, ids = rrs[subset], wqp[subset], ids[subset]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rrs))
    n_val = int(val_frac * len(rrs))
    va, tr = perm[:n_val], perm[n_val:]
    return wb, rrs[tr], wqp[tr], rrs[va], wqp[va]


def residual_modes(rrs, wavebands, rank=4):
    """Leading PCA modes of the deterministic hydropt fit residual.

    Used to initialise the low-rank noise covariance. Uses only Rrs -- no labels -- so it
    stays consistent with unsupervised training.
    """
    from  diagnose_misfit import fit_all
    _, model = fit_all(rrs, wavebands)
    r = (rrs - model) / rrs.mean(axis=1, keepdims=True)
    r = r[np.isfinite(r).all(axis=1)]
    _, s, vt = np.linalg.svd(r - r.mean(0), full_matrices=False)
    # scale each mode by its singular value so U U^T reproduces the observed covariance
    return vt[:rank] * (s[:rank] / np.sqrt(len(r)))[:, None]


def train(discrepancy='constants', epochs=400, seed=0, beta=1., lr=2e-3, quiet=False,
          noise='lowrank', noise_rank=4, subset=None):
    torch.manual_seed(seed)
    wb, rrs_tr, wqp_tr, rrs_va, wqp_va = load_data(seed, subset=subset)

    # Latent prior over plausible inland-water concentrations. Training never sees the
    # labels, so this is set from hydropt's own working range, not the training WQPs.
    # 0.5 decade per sd keeps +-3 sd inside ~3 decades -- wider than this and the
    # polynomial is evaluated where no real water lives.
    z_loc = np.log([5., .5, 5.])
    z_scale = np.log(10.) * np.array([.5, .5, .5])

    x_mean, x_std_ = rrs_tr.mean(0), rrs_tr.std(0) + 1e-9
    to_x = lambda r: torch.tensor((r - x_mean) / x_std_, dtype=DTYPE)
    Xtr, Rtr = to_x(rrs_tr), torch.tensor(rrs_tr, dtype=DTYPE)
    Xva, Rva = to_x(rrs_va), torch.tensor(rrs_va, dtype=DTYPE)

    modes = residual_modes(rrs_tr, wb, noise_rank) if noise == 'lowrank' else None
    model = PhysicsVAE(wb, z_loc, z_scale, discrepancy=discrepancy,
                       noise=noise, noise_rank=noise_rank, resid_modes=modes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n, bs = len(Xtr), 128

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for k in range(0, n, bs):
            b = idx[k:k + bs]
            nll, kl, _, _ = model(Xtr[b], Rtr[b])
            loss = (nll + beta * kl).mean()
            if discrepancy == 'residual':
                loss = loss + 1e2 * model.res_basis.pow(2).sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
        if not quiet and (ep + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                nll, kl, _, _ = model(Xva, Rva)
            print(f'  epoch {ep+1:4d}  val nll {nll.mean():9.1f}  kl {kl.mean():7.2f}')

    return model, (Xva, Rva, wqp_va), (Xtr, Rtr, wqp_tr)


def posterior(model, X, n_samples=512, seed=0):
    """Posterior samples of the concentrations. Returns (n_samples, n_spectra, 3)."""
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        mu, log_sd = model.encoder(X)
        z = mu[None] + torch.randn(n_samples, *mu.shape, dtype=mu.dtype) * log_sd.exp()[None]
        log_conc = (model.z_loc + model.z_scale * z).clamp(LOG_CONC_MIN, LOG_CONC_MAX)
    return log_conc.exp().numpy()


def reconstruct(model, X):
    """Decoder output at the posterior mean. Returns (rrs_hat, conc)."""
    model.eval()
    with torch.no_grad():
        mu, _ = model.encoder(X)
        rrs_hat, log_conc = model.decode(mu)
    return rrs_hat.numpy(), log_conc.exp().numpy()


def evaluate(model, data, n_samples=512, label=''):
    """Point skill and calibration of the posterior against in-situ values."""
    X, R, wqp = data
    conc = posterior(model, X, n_samples)                   # (S, B, 3)

    med = np.median(conc, axis=0)
    lo, hi = np.percentile(conc, [5, 95], axis=0)

    print(f'\n{label}')
    print(f'  {"WQP":<10}{"bias":>8}{"MdAPE":>8}{"RMSLE":>8}{"90% cover":>11}{"width":>9}')
    out = {}
    for j, name in enumerate(WQPS):
        o, e = wqp[:, j], med[:, j]
        m = np.isfinite(e) & (e > 0) & (o > 0)
        lr = np.log10(e[m] / o[m])
        cover = ((o >= lo[:, j]) & (o <= hi[:, j])).mean()
        width = np.median(np.log10(hi[:, j] / np.maximum(lo[:, j], 1e-12)))
        out[name] = dict(bias=10**np.median(lr),
                         mdape=100 * np.median(np.abs(e[m] - o[m]) / o[m]),
                         rmsle=float(np.sqrt(np.mean(lr**2))),
                         cover=float(cover), width=float(width))
        print(f'  {name:<10}{out[name]["bias"]:>8.2f}{out[name]["mdape"]:>7.0f}%'
              f'{out[name]["rmsle"]:>8.2f}{cover:>10.0%}{width:>9.2f}')
    print('  (cover = fraction of in-situ inside the 90% credible interval; '
          'width = log10 decades)')
    return out

def plot_input_vs_output(model, data, n_show=3, seed=0):
    X, R, _ = data
    rrs = R.numpy()
    rrs_hat, _ = reconstruct(model, X)
    wb = gloria.bands(hydropt_grid=True)

    rng = np.random.default_rng(seed)
    sub = rng.choice(len(rrs), min(n_show, len(rrs)), replace=False)

    fig, axes = plt.subplots(3, 2)
    for idx,i in enumerate(sub):
        axes[idx,0].plot(wb, rrs[i], color='0.7', lw=3.7, alpha=.8)
        axes[idx,1].plot(wb, rrs_hat[i], color='C0', lw=3.7, alpha=.55)
        axes[idx,0].set_axis_off()
        axes[idx,1].set_axis_off()

    # fig.suptitle('Spectral closure of the physics decoder')
    fig.tight_layout()
    return fig

def plot_retrieval(model, data, label='held-out', n_err=120, seed=0):
    """Retrieved vs in-situ, with 90% credible intervals on a subset of points.

    Error bars are drawn on a random subset only -- all ~360 would be unreadable, and
    the point of the panel is that they are far too short to reach the 1:1 line.
    """
    X, _, wqp = data
    conc = posterior(model, X)
    med = np.median(conc, axis=0)
    lo, hi = np.percentile(conc, [5, 95], axis=0)

    rng = np.random.default_rng(seed)
    sub = rng.choice(len(med), min(n_err, len(med)), replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, (j, name) in zip(axes, enumerate(WQPS)):
        o, e = wqp[:, j], med[:, j]
        m = np.isfinite(e) & (e > 0) & (o > 0)
        ax.errorbar(o[sub], med[sub, j],
                    yerr=[med[sub, j] - lo[sub, j], hi[sub, j] - med[sub, j]],
                    fmt='none', ecolor='0.75', elinewidth=.8, alpha=.7, zorder=1)
        ax.scatter(o[m], e[m], s=11, alpha=.6, edgecolor='none', zorder=2)
        lims = [min(o[m].min(), e[m].min()), max(o[m].max(), e[m].max())]
        ax.plot(lims, lims, 'k--', lw=1, zorder=3)

        cover = ((o >= lo[:, j]) & (o <= hi[:, j])).mean()
        lr = np.log10(e[m] / o[m])
        txt, unit = AXIS_LABEL[name]
        ax.set(xscale='log', yscale='log', xlim=lims, ylim=lims,
               xlabel=f'in-situ {txt} [{unit}]', ylabel=f'VAE posterior median [{unit}]')
        ax.set_title(f'{txt}   bias {10**np.median(lr):.2f}x, '
                     f'90% cov {cover:.0%}', fontsize=10)
    fig.suptitle(f'Physics-decoder VAE retrieval vs in-situ ({label}); '
                 'bars are 90% credible intervals')
    fig.tight_layout()
    return fig


def plot_closure(model, data, n_show=3, seed=0):
    """Spectral closure: does the physics decoder reproduce the measured spectrum?

    Left  -- measured (grey) with reconstruction (colour) for a random subset.
    Middle-- relative residual vs wavelength; the systematic shape here is the
             misspecification, and it is what makes the posterior overconfident.
    Right -- pooled per-band observed vs predicted.
    """
    X, R, _ = data
    rrs = R.numpy()
    rrs_hat, _ = reconstruct(model, X)
    wb = gloria.bands(hydropt_grid=True)

    rng = np.random.default_rng(seed)
    sub = rng.choice(len(rrs), min(n_show, len(rrs)), replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    ax = axes[0]
    # one colour per population, not per spectrum -- the point is how the two clouds
    # differ in shape, not how any individual pair lines up
    for i in sub:
        ax.plot(wb, rrs[i], color='0.7', lw=.7, alpha=.8)
        ax.plot(wb, rrs_hat[i], color='C0', lw=.7, alpha=.55)
    ax.plot([], [], color='0.7', lw=1.5, label='measured')
    ax.plot([], [], color='C0', lw=1.5, label='reconstructed')
    ax.legend(fontsize=8)
    ax.set(xlabel='wavelength [nm]', ylabel=r'$R_{rs}$ [1/sr]',
           title=f'measured vs reconstructed ({len(sub)} spectra)')

    ax = axes[1]
    rel = (rrs_hat - rrs) / rrs.mean(axis=1, keepdims=True)
    q = np.percentile(rel, [25, 50, 75], axis=0)
    ax.fill_between(wb, q[0], q[2], alpha=.25, label='IQR')
    ax.plot(wb, q[1], lw=1.5, label='median')
    ax.axhline(0, color='k', lw=1, ls='--')
    ax.set(xlabel='wavelength [nm]',
           ylabel=r'(model $-$ measured) / mean $R_{rs}$',
           title='residual')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.scatter(rrs.ravel(), rrs_hat.ravel(), s=2, alpha=.05, edgecolor='none')
    lim = [max(1e-5, np.percentile(rrs, .5)), rrs.max()]
    ax.plot(lim, lim, 'k--', lw=1)
    med_rel = np.median(np.linalg.norm(rrs - rrs_hat, axis=1)
                        / np.linalg.norm(rrs, axis=1))
    ax.set(xscale='log', yscale='log', xlim=lim, ylim=lim,
           xlabel=r'measured $R_{rs}$ [1/sr]', ylabel=r'model $R_{rs}$ [1/sr]',
           title=f'per-band closure, median $\\|r\\|/\\|R_{{rs}}\\|$ = {med_rel:.1%}')

    fig.suptitle('Spectral closure of the physics decoder')
    fig.tight_layout()
    return fig


def plot_latent_traversal(model, data=None, n_lines=9, span=2., seed=0):
    """What each latent does to the spectrum, and where its sensitivity vanishes.

    The physics decoder makes a latent traversal produce *spectra*, so this is the
    interpretable analogue of an image-VAE traversal grid. The bottom row is the more
    diagnostic half: d log Rrs / d log conc shows which bands actually carry information
    about each WQP, and a parameter with near-zero sensitivity everywhere is one the
    spectrum cannot constrain.
    """
    wb = gloria.bands(hydropt_grid=True)

    # anchor at the median posterior state, or the prior mean if no data given
    if data is not None:
        with torch.no_grad():
            mu, _ = model.encoder(data[0])
        anchor = mu.median(dim=0).values
    else:
        anchor = torch.zeros(3, dtype=DTYPE)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    offsets = np.linspace(-span, span, n_lines)

    for j, name in enumerate(WQPS):
        txt, unit = AXIS_LABEL[name]

        # ---- top: the spectral family -------------------------------------
        z = anchor.repeat(n_lines, 1).clone()
        z[:, j] = anchor[j] + torch.tensor(offsets, dtype=DTYPE)
        with torch.no_grad():
            rrs, log_conc = model.decode(z)
        conc = log_conc[:, j].exp().numpy()

        ax = axes[0, j]
        norm = plt.Normalize(np.log10(conc).min(), np.log10(conc).max())
        for k in range(n_lines):
            ax.plot(wb, rrs[k].numpy(), color=plt.cm.viridis(norm(np.log10(conc[k]))),
                    lw=1.3)
        ax.set(xlabel='wavelength [nm]', ylabel=r'$R_{rs}$ [1/sr]', title=txt)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax,
                     label=rf'$\log_{{10}}$ {txt} [{unit}]')

        # ---- bottom: relative sensitivity at the anchor ---------------------
        # d log Rrs / d log conc, via autograd on the exact physics decoder
        za = anchor.clone().unsqueeze(0).requires_grad_(True)
        out, _ = model.decode(za)
        sens = np.stack([
            torch.autograd.grad(out[0, b], za, retain_graph=True)[0][0].numpy()
            for b in range(out.shape[1])])                  # (bands, 3), d Rrs / d z
        rrs0 = out.detach().numpy()[0]
        # z is standardised log-conc, so d/dz = z_scale * d/d log conc
        dlog = sens[:, j] / rrs0 / model.z_scale[j].item()

        ax = axes[1, j]
        ax.plot(wb, dlog, lw=1.5, color=f'C{j}')
        ax.axhline(0, color='k', lw=.8, ls='--')
        ax.set(xlabel='wavelength [nm]',
               ylabel=r'$\partial \log R_{rs} / \partial \log$ conc',
               title=f'{txt}: spectral sensitivity')

    fig.suptitle('Latent traversal through the physics decoder '
                 '(top: spectra, bottom: sensitivity at the anchor)')
    fig.tight_layout()
    return fig


def lm_estimate(model, rrs_row, wavebands=None):
    """Deterministic Levenberg-Marquardt fit of one spectrum, for overlay.

    Uses the model's *learned* constants so the comparison is against the same physics
    the VAE sees. Defined here rather than imported from stage3_compare, which imports
    this module.
    """
    import  hydropt_flex as hf
    wb = gloria.bands(hydropt_grid=True) if wavebands is None else wavebands
    const = {nm: model.decoder.constant(nm).item() for nm in model.decoder.learn}
    _, _, inv = hf.build(wb, **const)
    try:
        h = inv.invert(y=np.asarray(rrs_row), x=hf.lmfit_params((.01, .01, .01)), jac=True)
        return np.array([h.params[p].value for p in ('phyto', 'cdom', 'nap')])
    except Exception:
        return np.full(3, np.nan)


def plot_degeneracy(model, data, indices=None, n_grid=81, n_cdom=25, span=4.,
                    n_post=400, chunk=20000, seed=0):
    """Profiled likelihood over (chl, TSS) with the posterior drawn on top.

    CDOM is profiled out rather than fixed -- that is what makes the valley honest. The
    contoured quantity is the model's own NLL (including the correlated noise term), so
    the surface is in the units the VAE actually optimises.

    The mean-field posterior is axis-aligned by construction (Encoder returns a diagonal
    log_sd), so if the valley is a tilted ridge the posterior cloud structurally cannot
    follow it. That mismatch is the point of the figure.
    """
    X, R, wqp = data
    rrs = R.numpy()

    if indices is None:                       # best / median / worst by chl log-error
        med = np.median(posterior(model, X), axis=0)
        err = np.abs(np.log10(np.maximum(med[:, 0], 1e-9) / np.maximum(wqp[:, 0], 1e-9)))
        order = np.argsort(err)
        indices = [order[0], order[len(order) // 2], order[-1]]
    indices = list(indices)

    # grid in standardised latent space, converted to physical units for plotting
    g = torch.tensor(np.linspace(-span, span, n_grid), dtype=DTYPE)
    c = torch.tensor(np.linspace(-span, span, n_cdom), dtype=DTYPE)
    ZC, ZN, ZD = torch.meshgrid(g, g, c, indexing='ij')
    Z = torch.stack([ZC.reshape(-1), ZD.reshape(-1), ZN.reshape(-1)], dim=1)  # chl,cdom,nap

    fig, axes = plt.subplots(1, len(indices), figsize=(5.2 * len(indices), 4.6))
    axes = np.atleast_1d(axes)

    for ax, i in zip(axes, indices):
        target = torch.tensor(rrs[i], dtype=DTYPE).unsqueeze(0)
        nll = torch.empty(len(Z), dtype=DTYPE)
        with torch.no_grad():
            for s in range(0, len(Z), chunk):
                zb = Z[s:s + chunk]
                rhat, _ = model.decode(zb)
                nll[s:s + chunk] = model._nll(target.expand(len(zb), -1), rhat)
        # profile out CDOM: keep the best value at each (chl, nap) node
        surf = nll.reshape(n_grid, n_grid, n_cdom).min(dim=2).values.numpy()
        surf = surf - surf.min()

        # axes in physical log10 units
        chl_ax = (model.z_loc[0].item() + model.z_scale[0].item() * g.numpy()) / np.log(10)
        nap_ax = (model.z_loc[2].item() + model.z_scale[2].item() * g.numpy()) / np.log(10)

        # cap at 25 nats: past that everything is equally rejected and extra range only
        # saturates the panel, hiding the valley that is the whole point
        levels = [.25, .5, 1, 2, 5, 10, 25]
        cs = ax.contourf(chl_ax, nap_ax, surf.T, levels=[0] + levels,
                         cmap='magma_r', extend='max', alpha=.9)
        ax.contour(chl_ax, nap_ax, surf.T, levels=[1.], colors='w', linewidths=1.8)

        # overlays, all in log10 physical units
        post = posterior(model, X[i:i + 1], n_samples=n_post, seed=seed)[:, 0, :]
        ax.scatter(np.log10(post[:, 0]), np.log10(post[:, 2]), s=4, alpha=.35,
                   color='#39FF14', edgecolor='none', label='VAE posterior', zorder=4)
        lm = lm_estimate(model, rrs[i])
        if np.isfinite(lm).all():
            ax.plot(np.log10(max(lm[0], 1e-9)), np.log10(max(lm[2], 1e-9)), 'w^',
                    ms=9, mec='k', label='LM estimate', zorder=5)
        ax.plot(np.log10(wqp[i, 0]), np.log10(wqp[i, 2]), 'r*', ms=16, mec='k',
                label='in-situ truth', zorder=6)

        ax.set(xlabel=r'$\log_{10}$ chl-a [$\mu$g/L]',
               xlim=(chl_ax.min(), chl_ax.max()), ylim=(nap_ax.min(), nap_ax.max()))
        if ax is axes[0]:                      # shared y axis; label once
            ax.set_ylabel(r'$\log_{10}$ TSS [mg/L]')
        in_set = surf[np.argmin(np.abs(chl_ax - np.log10(wqp[i, 0]))),
                      np.argmin(np.abs(nap_ax - np.log10(wqp[i, 2])))]
        ax.set_title(f'spectrum {i}   truth at $\\Delta$={in_set:.0f} nats', fontsize=10)
        ax.legend(fontsize=7, loc='upper right', framealpha=.85)

    fig.colorbar(cs, ax=axes.tolist(), label=r'$\Delta(-\log L)$ [nats], CDOM profiled out')
    fig.suptitle('Likelihood degeneracy vs the mean-field posterior '
                 '(white contour = 1 nat, the indistinguishable set)')
    return fig


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--discrepancy', default='constants',
                    choices=['none', 'constants', 'residual'])
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--beta', type=float, default=1.)
    ap.add_argument('--noise', default='lowrank', choices=['diag', 'lowrank'])
    ap.add_argument('--noise-rank', type=int, default=4)
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--degeneracy', action='store_true',
                    help='also draw the profiled-likelihood degeneracy map (slow)')
    # parse_known_args, not parse_args: a Jupyter kernel injects its own
    # --f=<connection file>, and argparse would sys.exit(2) on it. Running this file in
    # an interactive window should just train with the defaults.
    args, _ = ap.parse_known_args(argv)
    return args


# %% run

# def run(args=None, **overrides):
"""Train and report. Callable from a cell as run(epochs=100) or from the CLI."""
# args = args or parse_args()
# for k, v in overrides.items():
#     setattr(args, k, v)
args = parse_args()

print(f'physics-decoder VAE, discrepancy={args.discrepancy}, '
        f'noise={args.noise}, beta={args.beta}')
model, val, tr = train(args.discrepancy, epochs=args.epochs, beta=args.beta,
                        noise=args.noise, noise_rank=args.noise_rank)
evaluate(model, tr, label='train')
evaluate(model, val, label='held-out')

if args.discrepancy != 'none':
    print('\nlearned constants (hydropt default in brackets):')
    import  hydropt_flex as hf
    for nm in model.decoder.learn:
        print(f'  {nm:<16}{model.decoder.constant(nm).item():.5f}  '
                f'[{hf.DEFAULTS[nm]:.5f}]')
#%% plot
if not getattr(args, 'no_plots', False):
    fig0 = plot_retrieval(model, val, label='held-out')
    fig1 = plot_closure(model, val)
    fig2 = plot_latent_traversal(model, val)
    fig3 = plot_input_vs_output(model, val)
    if save_figures:
        fig0.savefig(f'{fig_folder}/vae_retrievals.png', dpi=300)
        fig1.savefig(f'{fig_folder}/vae_closure.png', dpi=300)
        fig2.savefig(f'{fig_folder}/vae_latent_traversal.png', dpi=300)
        fig3.savefig(f'{fig_folder}/vae_input_vs_output.png', dpi=300)
    if getattr(args, 'degeneracy', False):     # slow: opt in
        plot_degeneracy(model, val)
    plt.show()
# return model, val, tr


# %% run
# if __name__ == '__main__':
    # model, val, tr = run()
