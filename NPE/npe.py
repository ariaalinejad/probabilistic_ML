"""Neural Posterior Estimation for GLORIA water quality retrieval.

q(theta | x) is a conditional normalizing flow over theta = log10 of the four WQPs,
conditioned on the 6-D EDAA feature vector from features.py. Trained on real matchups,
so unlike the physics-decoder VAE there is no forward model to be misspecified, and
Secchi depth -- which has no HYDROPT state variable -- is retrievable like the rest.

This is NPE without a simulator. The usual SBI setup draws (theta, x) from a prior and a
forward model; here the 942 GLORIA matchups *are* the joint sample. That is the whole
point: the joint includes GLORIA's real label noise and real water types, neither of
which HYDROPT reproduces. The cost is that nothing can be simulated outside the data, so
training is offline over a fixed set and the sample size is the binding constraint.

A flow rather than a Gaussian posterior because the labels are strongly dependent --
log10 Secchi against log10 TSS correlates at -0.93. A diagonal-Gaussian posterior (the
VAE's mean-field encoder) structurally cannot represent a tilted ridge like that; a
coupling flow can.
"""
import argparse
import os

# Must precede the bayesflow import: ~/.keras/keras.json requests tensorflow, which is
# not installed in this venv, so the default backend would fail at import time.
os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np                                                   # noqa: E402
import keras                                                         # noqa: E402
import bayesflow as bf                                               # noqa: E402

import data                                                          # noqa: E402
from probabilistic_ml.NPE.features import SpectralFeatures, N_COMPONENTS                  # noqa: E402

WQPS = data.WQPS


def build_workflow(depth=4, width=128, transform='spline', network='coupling'):
    """A conditional flow over theta, conditioned on the feature vector.

    No summary network: the observation is already a fixed-length 6-vector, so the
    features go straight to inference_conditions.

    Kept small deliberately. With ~750 training spectra per fold the risk is an
    overconfident posterior, not an underexpressive one.
    """
    adapter = (bf.adapters.Adapter()
               .to_array()
               .convert_dtype('float64', 'float32')
               .rename('theta', 'inference_variables')
               .rename('features', 'inference_conditions'))

    subnet_kwargs = dict(widths=(width, width))
    if network == 'flow_matching':
        inference_network = bf.networks.FlowMatching(subnet_kwargs=subnet_kwargs)
    else:
        inference_network = bf.networks.CouplingFlow(
            depth=depth, transform=transform, subnet_kwargs=subnet_kwargs)

    return bf.BasicWorkflow(
        adapter=adapter,
        inference_network=inference_network,
        # standardising both sides is what lets the flow start from a sane scale;
        # theta is log10 WQPs spanning several decades
        standardize=['inference_variables', 'inference_conditions'],
    )


def as_dataset(theta, features):
    """BayesFlow wants a dict of arrays keyed by the names the adapter renames."""
    return {'theta': np.asarray(theta, dtype=np.float64),
            'features': np.asarray(features, dtype=np.float64)}


def train_fold(rrs, theta, groups, fold=0, n_splits=5, epochs=300, batch_size=64,
               seed=0, p=N_COMPONENTS, use_magnitude=True, quiet=False, **net_kwargs):
    """Fit features and flow on one grouped fold.

    The feature extractor is fit on the training rows only and frozen before the test
    rows are transformed -- otherwise the test spectra shape the representation and the
    calibration numbers are contaminated.

    Returns (workflow, extractor, test_data, history) where test_data is the dict the
    workflow samples from.
    """
    train_idx, test_idx = data.split(groups, fold=fold, n_splits=n_splits)
    inner_train, inner_val = data.holdout(groups[train_idx], seed=seed)
    # holdout indexes into train_idx, so lift back to positions in the full arrays
    fit_idx, val_idx = train_idx[inner_train], train_idx[inner_val]

    keras.utils.set_random_seed(seed)

    extractor = SpectralFeatures(p=p, seed=seed, use_magnitude=use_magnitude).fit(rrs[fit_idx])
    train_set = as_dataset(theta[fit_idx], extractor.transform(rrs[fit_idx]))
    val_set = as_dataset(theta[val_idx], extractor.transform(rrs[val_idx]))
    test_set = as_dataset(theta[test_idx], extractor.transform(rrs[test_idx]))

    workflow = build_workflow(**net_kwargs)
    history = workflow.fit_offline(
        data=train_set, epochs=epochs, batch_size=batch_size,
        validation_data=val_set, verbose=0 if quiet else 1)

    if not quiet:
        losses = history.history
        final = {k: v[-1] for k, v in losses.items() if 'loss' in k}
        print(f'fold {fold}: train {len(fit_idx)} / val {len(val_idx)} / '
              f'test {len(test_idx)}  ' +
              '  '.join(f'{k}={v:.3f}' for k, v in final.items()))

    return workflow, extractor, test_set, test_idx, history


def sample_posterior(workflow, dataset, num_samples=1000, batch_size=64):
    """(num_samples, n, k) posterior draws in log10 WQP space."""
    conditions = {'features': dataset['features']}
    drawn = workflow.sample(conditions=conditions, num_samples=num_samples,
                            batch_size=batch_size)
    # BayesFlow returns (n, num_samples, k); put samples first to match the VAE's layout
    theta = drawn['theta'] if isinstance(drawn, dict) else drawn
    return np.asarray(theta).transpose(1, 0, 2)


def main():
    # allow_abbrev=False so a Jupyter kernel's -f <connection file> is not matched as
    # an abbreviation of --folds; parse_known_args alone does not save us, because an
    # abbreviated flag counts as *recognised* and then fails on int('...kernel.json')
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument('--folds', type=int, default=1, help='how many CV folds to run')
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--network', choices=['coupling', 'flow_matching'], default='coupling')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--quiet', action='store_true')
    # a Jupyter kernel injects --f=<connection file>; argparse would exit(2) on it
    args, _ = ap.parse_known_args()

    rrs, theta, ids, groups, meta = data.load()
    print(f'{len(rrs)} matchups, {len(np.unique(groups))} campaigns, '
          f'backend={keras.backend.backend()}\n')

    for fold in range(args.folds):
        workflow, extractor, test_set, _, _ = train_fold(
            rrs, theta, groups, fold=fold, n_splits=args.n_splits,
            epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
            quiet=args.quiet, depth=args.depth, width=args.width,
            network=args.network)

        draws = sample_posterior(workflow, test_set, num_samples=500)
        median = np.median(draws, axis=0)
        truth = test_set['theta']
        print(f'  {"WQP":<15}{"RMSLE":>8}{"r_log":>8}')
        for j, col in enumerate(WQPS):
            rmsle = float(np.sqrt(np.mean((median[:, j] - truth[:, j]) ** 2)))
            r = float(np.corrcoef(truth[:, j], median[:, j])[0, 1])
            print(f'  {col:<15}{rmsle:8.3f}{r:8.3f}')


if __name__ == '__main__':
    main()
