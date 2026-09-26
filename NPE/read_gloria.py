# %% imports
import matplotlib.pyplot as plt
import numpy as np
import hydropt.hydropt as hd
from hydropt.bio_optics import H2O_IOP_DEFAULT
from hydropt.bio_optics import a_phyto_base_HSI
import lmfit

import probabilistic_ml.NPE.gloria as gloria
# %% setup and definitions

gloria_data_dir = gloria.GLORIA_DIR

# the four WQPs of interest. Units: ug/L, mg/L, 1/m, m
wqp_columns = gloria.WQP_COLUMNS

# %% functions

# The loading and matchup logic now lives in gloria.py so the diagnostics and the VAE
# share it rather than re-reading ~40 MB of CSV each. get_matchups stays as a thin alias
# so the exploratory cells below read the same as before.
get_matchups = gloria.matchups

# %% read data / joint table

dataset, clean, rrs_bands = gloria.load()   # rrs_bands: 350-900 nm, 1 nm steps

print(f'{len(dataset)} spectra, {len(clean)} pass QC')
for col in wqp_columns:
    print(f'  {col:<13} {clean[col].notna().sum():>5} labelled')
print(f'  {"all four":<13} {clean[wqp_columns].notna().all(axis=1).sum():>5} labelled')


# %% plot spectra coloured by chl-a

rrs, labels, _ = get_matchups(['Chla'])

rng = np.random.default_rng(0)
subset = rng.choice(len(rrs), size=300, replace=False)
colours = np.log10(labels[subset, 0])

fig, ax = plt.subplots(figsize=(8, 5))
norm = plt.Normalize(colours.min(), colours.max())
for i, c in zip(subset, colours):
    ax.plot(rrs_bands, rrs[i], color=plt.cm.viridis(norm(c)), lw=0.5, alpha=0.6)

ax.set_xlabel('wavelength [nm]')
ax.set_ylabel(r'$R_{rs}$ [1/sr]')
ax.set_title(f'GLORIA: {len(rrs)} spectra with chl-a (300 shown)')
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax,
             label=r'$\log_{10}$ chl-a [$\mu$g/L]')
plt.show()

# plot mean of top 30 spectra with highest wqp value
top_n = 30
fig, ax = plt.subplots(figsize=(8, 5))
for wqp in wqp_columns:
    rrs, labels, _ = get_matchups([wqp])
    top_idx = np.argsort(labels[:, 0])[-top_n:]
    mean_rrs = rrs[top_idx].mean(axis=0)

    ax.plot(rrs_bands, mean_rrs)
    ax.set_xlabel('wavelength [nm]')
    ax.set_ylabel(r'$R_{rs}$ [1/sr]')
    ax.set_title(f'GLORIA: mean of {top_n} spectra with highest {wqp}')
plt.legend(wqp_columns)
plt.show()

# %% test hydropt

wavebands = np.arange(400, 711, 5)

def clear_nat_water(*args):
    return H2O_IOP_DEFAULT.T.values

def phytoplankton(*args):
    chl = args[0]
    # basis vector - according to Ciotti&Cullen (2002)
    a = a_phyto_base_HSI.absorption.values
    # constant spectral backscatter with backscatter ratio of 1.4%
    bb = np.repeat(.014*0.18, len(a))

    return chl*np.array([a, bb])

def cdom(*args):
    # absorption at 440 nm
    a_440 = args[0]
    # spectral absorption
    a = np.array(np.exp(-0.017*(wavebands-440)))
    # no backscatter
    bb = np.zeros(len(a))

    return a_440*np.array([a, bb])

bio_opt = hd.BioOpticalModel()
# set optical models
bio_opt.set_iop(
    wavebands=wavebands,
    water=clear_nat_water,
    phyto=phytoplankton,
    cdom=cdom)

bio_opt.plot(water=None, phyto=1, cdom=1)

# the HYDROPT polynomial forward model
fwd_model = hd.PolynomialForward(bio_opt)
# calculate Rrs
rrs = fwd_model.forward(phyto=.15, cdom=.02)

plt.figure()
plt.plot(wavebands, rrs)
plt.xlabel('wavelength [nm]')
plt.ylabel(r'$R_{rs}$ [1/sr]')


# set initial guess parameters for LM
x0 = lmfit.Parameters()
# some initial guess
x0.add('phyto', value=.5,  min=1E-9)
x0.add('cdom', value=.01, min=1E-9)

# initialize an inversion model
inv_model = hd.InversionModel(
    fwd_model=fwd_model,
    minimizer=lmfit.minimize)
# estimate parameters
xhat = inv_model.invert(y=rrs, x=x0)


# %% comparing hydopt with the GLORIA data


# %% matchups on the hydropt grid

from functools import partial
from hydropt.bio_optics import clear_nat_water as h2o_iop, phyto as phyto_iop
from hydropt.bio_optics import cdom as cdom_iop, nap as nap_iop

# hydropt runs 400-710 nm at 5 nm; GLORIA is 350-900 at 1 nm, so subset rather than interpolate
band_idx = np.searchsorted(rrs_bands, wavebands)
assert np.array_equal(rrs_bands[band_idx], wavebands)

# hydropt's state variables line up with GLORIA's columns and units 1:1:
#   phyto -> Chla [ug/L],  cdom -> aCDOM440 [1/m],  nap -> TSS [mg/L]
rrs_gl, wqp_gl, ids_gl = get_matchups(['Chla', 'aCDOM440', 'TSS'])
rrs_gl = rrs_gl[:, band_idx]

# a few spectra have gaps inside the hydropt window
ok = np.isfinite(rrs_gl).all(axis=1)
rrs_gl, wqp_gl, ids_gl = rrs_gl[ok], wqp_gl[ok], ids_gl[ok]
print(f'{ok.sum()} matchups with all 3 WQPs and no gaps in 400-710 nm')

# %% two forward models to compare
#
# The tutorial model above has no NAP, and in GLORIA's mostly-turbid inland waters that
# is fatal: it tops out around 5e-4 /sr while GLORIA's median Rrs is ~4.7e-3 /sr. Adding
# phyto only darkens the spectrum (absorption outruns its weak backscatter), so the
# optimiser drives chl to the lower bound trying to brighten the fit. Mineral backscatter
# is what actually makes these waters bright, so NAP is not optional here.

def build(**components):
    bio = hd.BioOpticalModel()
    bio.set_iop(wavebands=wavebands, water=h2o_iop, **components)
    fwd = hd.PolynomialForward(bio)
    return hd.InversionModel(fwd_model=fwd, minimizer=lmfit.minimize)

models = {
    'phyto+cdom':     (build(phyto=phyto_iop,
                             cdom=partial(cdom_iop, wb=wavebands)),
                       ['phyto', 'cdom']),
    'phyto+cdom+nap': (build(phyto=phyto_iop,
                             cdom=partial(cdom_iop, wb=wavebands),
                             nap=partial(nap_iop, wb=wavebands)),
                       ['phyto', 'cdom', 'nap']),
}

# %% invert

n_invert = 400
sel = np.random.default_rng(0).choice(len(rrs_gl), n_invert, replace=False)
measured = wqp_gl[sel]
start = {'phyto': 1., 'cdom': .01, 'nap': 1.}

results = {}
for name, (inv, params) in models.items():
    x0 = lmfit.Parameters()
    for p in params:
        x0.add(p, value=start[p], min=1E-9)

    out = np.full((n_invert, len(params)), np.nan)
    for k, i in enumerate(sel):
        try:                                # jac=True uses hydropt's analytical gradient
            xhat = inv.invert(y=rrs_gl[i], x=x0, jac=True)
            out[k] = [xhat.params[p].value for p in params]
        except Exception:                   # non-convergence on the odd spectrum
            pass
    results[name] = dict(zip(params, out.T))

# %% retrieval skill

def skill(obs, est):
    """Log-space metrics: WQPs span orders of magnitude, so ratios beat differences."""
    m = np.isfinite(est) & (est > 1E-6) & np.isfinite(obs) & (obs > 0)
    if m.sum() < 3:
        return f'  n={m.sum():4d}  (degenerate)'
    o, e = obs[m], est[m]
    log_ratio = np.log10(e / o)
    return (f'  n={m.sum():4d}  '
            f'bias={10**np.median(log_ratio):6.2f}x  '
            f'MdAPE={100*np.median(np.abs(e - o) / o):5.0f}%  '
            f'RMSLE={np.sqrt(np.mean(log_ratio**2)):5.2f}  '
            f'r_log={np.corrcoef(np.log10(o), np.log10(e))[0, 1]:5.2f}')

# n counts only retrievals that stayed off the lower bound -- a model that collapses
# chl to ~0 looks unbiased on the survivors, so read n alongside the metrics
wqp_of = {'phyto': ('Chla', 0), 'cdom': ('aCDOM440', 1), 'nap': ('TSS', 2)}
print(f'\nhydropt inversion vs GLORIA in-situ, n={n_invert} spectra')
for name, retr in results.items():
    print(f'\n{name}')
    for p, est in retr.items():
        col, j = wqp_of[p]
        print(f'  {col:<9}{skill(measured[:, j], est)}')

# %% scatter: retrieved vs in-situ

axis_label = {'phyto': ('chl-a', r'$\mu$g/L'),
              'cdom': (r'$a_{CDOM}(440)$', '1/m'),
              'nap': ('TSS', 'mg/L')}

fig, axes = plt.subplots(len(models), 3, figsize=(13, 8))
for row, (name, retr) in zip(np.atleast_2d(axes), results.items()):
    for ax, p in zip(row, ['phyto', 'cdom', 'nap']):
        if p not in retr:
            ax.axis('off')
            continue
        col, j = wqp_of[p]
        o, e = measured[:, j], retr[p]
        m = np.isfinite(e) & (e > 1E-6) & (o > 0)
        label, unit = axis_label[p]
        ax.scatter(o[m], e[m], s=10, alpha=.45, edgecolor='none')
        lims = [min(o[m].min(), e[m].min()), max(o[m].max(), e[m].max())]
        ax.plot(lims, lims, 'k--', lw=1)
        ax.set(xscale='log', yscale='log', xlim=lims, ylim=lims,
               xlabel=f'in-situ [{unit}]', ylabel=f'hydropt [{unit}]')
        # how many hit the bound and dropped out of the comparison
        ax.set_title(f'{name}: {label}  ({m.sum()}/{n_invert})', fontsize=10)
fig.suptitle('HYDROPT inversion vs GLORIA in-situ')
fig.tight_layout()
plt.show()

# %% forward simulation check (closure test)
#
# Simulate Rrs from the 3-component model itself, then invert it back. This removes
# model-data mismatch entirely, so whatever error survives is intrinsic to the forward
# operator + optimiser -- it separates "hydropt cannot represent GLORIA" from
# "chl and NAP are inherently degenerate".

inv_sim = models['phyto+cdom+nap'][0]
fwd_sim = inv_sim._fwd_model
PARAMS = ['phyto', 'cdom', 'nap']

def lmfit_params(values, names=PARAMS):
    p = lmfit.Parameters()
    for k, v in zip(names, values):
        p.add(k, value=v, min=1E-9)
    return p

n_sim = 400
rng_sim = np.random.default_rng(1)

# sample truths log-uniform over the range GLORIA actually spans (columns already
# ordered Chla, aCDOM440, TSS == phyto, cdom, nap)
lo, hi = np.percentile(wqp_gl, [2, 98], axis=0)
truth = 10 ** rng_sim.uniform(np.log10(lo), np.log10(hi), size=(n_sim, 3))
rrs_sim = np.array([fwd_sim.forward(**dict(zip(PARAMS, t))) for t in truth])

print(f'\nclosure test: {n_sim} spectra simulated over GLORIA-like ranges')
for p, a, b in zip(PARAMS, lo, hi):
    print(f'  {p:<6} {a:8.3f} .. {b:8.2f}')
print(f'  simulated Rrs median {np.median(rrs_sim):.4f} /sr '
      f'vs GLORIA {np.median(rrs_gl):.4f} /sr')

# additive white noise as a fraction of each spectrum's own mean Rrs
noise_levels = {'noise-free': 0.0, '1% of mean': .01, '5% of mean': .05, '12% of mean': .12}

closure = {}
for lab, sigma in noise_levels.items():
    y = rrs_sim + (rng_sim.normal(0, sigma * rrs_sim.mean(axis=1, keepdims=True),
                                  rrs_sim.shape) if sigma else 0.)
    out = np.full((n_sim, 3), np.nan)
    for k in range(n_sim):
        try:                                # start low: see stability note below
            h = inv_sim.invert(y=y[k], x=lmfit_params((.01, .01, .01)), jac=True)
            out[k] = [h.params[p].value for p in PARAMS]
        except Exception:
            pass
    closure[lab] = out
    print(f'\n{lab}')
    for j, p in enumerate(PARAMS):
        print(f'  {p:<6}{skill(truth[:, j], out[:, j])}')

# %% closure vs real data, side by side

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (j, p) in zip(axes, enumerate(PARAMS)):
    for lab in ['noise-free', '5% of mean', '12% of mean']:
        e = closure[lab][:, j]
        m = np.isfinite(e) & (e > 1E-6)
        ax.scatter(truth[m, j], e[m], s=8, alpha=.4, edgecolor='none', label=lab)
    lims = [truth[:, j].min(), truth[:, j].max()]
    ax.plot(lims, lims, 'k--', lw=1)
    label, unit = axis_label[p]
    ax.set(xscale='log', yscale='log', xlabel=f'true [{unit}]',
           ylabel=f'retrieved [{unit}]', title=f'closure: {label}')
    ax.legend(fontsize=7)
fig.suptitle('HYDROPT closure test -- data generated by the model itself')
fig.tight_layout()
plt.show()

# %% if I make a main()

# try:
#     _IPYTHON_
#     main()
# except NameError:
#     if __name__ == '__main__':
#         main()