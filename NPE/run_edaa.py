# %% imports
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr

import config
import interpret_endmembers as ie
import visualize_owt as vo
from probabilistic_ml.NPE.edaa import BlindEDAA
from spectra_selection import (band_mask, filter_valid_spectra, normalize,
                               select_spectra)

# %% definitions and configurations

# everything shared with the rest of the pipeline lives in config.py
n_scenes = config.n_scenes
n_components = config.n_components
num_pixels_per_scene = config.num_pixels_per_scene
min_usable_spectra = config.min_usable_spectra
selection_seed = config.selection_seed
spectra_var = config.spectra_var
training_reject_flags = config.training_reject_flags
band_range = config.band_range
swir_thresh = config.swir_thresh
binning_factor = config.binning_factor
endmember_path = config.endmember_path
data_path = config.product_path(config.product)

# False reuses the saved endmembers instead of re-fitting. The load is refused if
# the file does not match the current settings, so a stale file cannot silently
# produce wrong maps.
recompute_endmembers = True

# write every figure to config.figure_dir as well as showing it
save_figures = True

np.random.seed(config.random_seed)
# %% temp functions

column_width_in = config.column_width_in

def archetype_names(p, labels=None):
    '''
    Display names for p archetypes: an explicit `labels`, else the poster names
    in config, else AA_1 ... AA_p.

    Centralised because a figure titled "AA_2" and a caption calling the same
    thing "clear water" is how a poster loses its reader. config.archetype_names
    is only used when its length matches the fit, so changing n_components
    cannot silently mislabel every panel.
    '''
    if labels is not None:
        return list(labels)
    configured = getattr(config, 'archetype_names', None)
    if configured and len(configured) == p:
        return list(configured)
    return [f'AA_{k + 1}' for k in range(p)]


def _names(results, labels):
    '''Archetype names for a results list.'''
    return archetype_names(results[0]['abundances'].shape[0], labels)


def _scene_means(results):
    '''
    Mean abundance per scene (n, p), raw and scaled by each archetype's maximum,
    plus the polar angles and their closed form for the radar plots.

    Scaling per axis is what makes the radars readable: raw mean abundances all
    sit near 1/p and the polygons would be indistinguishable. After scaling, 1.0
    means "the scene where this water type was most prevalent".
    '''
    p = results[0]['abundances'].shape[0]
    means = np.array([r['abundances'].mean(axis=1) for r in results])
    scaled = means / means.max(axis=0, keepdims=True)
    angles = np.linspace(0, 2 * np.pi, p, endpoint=False)
    return means, scaled, angles, np.concatenate([angles, angles[:1]])

def plot_scene_small_multiples(results, labels=None):
    '''
    One small radar per scene, since ten overlaid polygons are unreadable.

    The dashed grey reference is the mean across scenes, so a scene's departure
    from typical composition is visible without cross-referencing the others.
    '''
    names = _names(results, labels)
    _, scaled, angles, closed = _scene_means(results)
    colours = plt.cm.turbo(np.linspace(0, 1, len(results)))

    columns = min(5, len(results))
    rows = -(-len(results) // columns)
    fig, axes = plt.subplots(rows, columns,
                             figsize=(column_width_in,
                                      0.25 * column_width_in * rows),
                             subplot_kw={'projection': 'polar'}, squeeze=False,
                             layout='constrained')
    reference = np.concatenate([scaled.mean(axis=0), scaled.mean(axis=0)[:1]])
    for i, result in enumerate(results):
        axis = axes[i // columns, i % columns]
        axis.plot(closed, reference, color='0.75', lw=1, ls='--')
        values = np.concatenate([scaled[i], scaled[i, :1]])
        axis.plot(closed, values, color=colours[i], lw=1.8)
        axis.fill(closed, values, color=colours[i], alpha=0.25)
        axis.set_title(result['label'], fontsize=15, pad=14)
        # only the AA-n prefix fits round a small radar; the full name is in the
        # legend of the archetype-spectra panel
        axis.set_xticks(angles, [n.split()[0] for n in names], fontsize=11)
        axis.set_yticklabels([])
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.4)
    for i in range(len(results), rows * columns):
        axes[i // columns, i % columns].axis('off')
    fig.suptitle('Scene compositions')
    fig.tight_layout()
    return fig
# %% functions

def flag_bits(flags_da):
    """Map l2_flag name -> bit value, from the field's own self-describing attrs."""
    return dict(zip(flags_da.attrs['flag_meanings'].split(),
                    np.asarray(flags_da.attrs['flag_masks'])))


def flag_reject_mask(ds, reject):
    """
    Flat (n_pixels,) bool, True where *none* of `reject` is set.

    All-True if the scene carries no l2_flags, so an older file gets no extra
    filtering rather than losing every pixel.
    """
    if 'l2_flags' not in ds or not reject:
        n_lines, n_pixels = ds[spectra_var].shape[:2]
        return np.ones(n_lines * n_pixels, dtype=bool)

    flags = ds['l2_flags']
    name_to_bit = flag_bits(flags)
    bits = 0
    for name in reject:
        if name in name_to_bit:
            bits |= int(name_to_bit[name])
    return (flags.values.ravel() & bits) == 0


def bin_bands(X, wavelengths, factor=binning_factor):
    '''
    Average adjacent bands in groups of `factor`, over the last axis of `X`.

    Averaging raises SNR at the cost of spectral resolution, and shrinks the
    problem EDAA solves. Trailing bands that do not fill a whole group are
    dropped so the spectra and the band grid stay aligned. A factor of 1 (or
    None) returns both unchanged.

    Applied identically to the training spectra and to every scene unmixed
    later -- the endmembers only exist on the binned grid, so `transform()`
    would fail on unbinned pixels.

    Returns (binned X, binned wavelengths).
    '''
    if not factor or factor <= 1:
        return X, np.asarray(wavelengths)
    n = (X.shape[-1] // factor) * factor
    binned = X[..., :n].reshape(*X.shape[:-1], n // factor, factor).mean(axis=-1)
    return binned, np.asarray(wavelengths)[:n].reshape(-1, factor).mean(axis=1)


def _scatter(values, valid_indices, shape):
    '''
    Put per-pixel `values`, (n_valid,) or (n_valid, k), back where they came
    from; everything else NaN.
    '''
    trailing = values.shape[1:]
    flat = np.full((shape[0] * shape[1], *trailing), np.nan)
    flat[valid_indices] = values
    return flat.reshape(*shape, *trailing)


def unmix_scene(filepath, model, endmembers):
    '''
    Apply already-fit endmembers to every pixel in a scene with a usable
    spectrum, returning both the spatial maps and the flat per-pixel arrays.

    The keys match what visualize_owt's plot functions expect. Returns None if
    the scene has no usable pixels.
    '''
    pace_xr = xr.open_dataset(filepath)
    wavelengths = pace_xr['wavelength_3d'].values
    pace_spectra_flat = pace_xr[spectra_var].values.reshape(-1, len(wavelengths))

    # Deliberately *not* the training filter: the quality-flagged pixels held out
    # of the fit are unmixed here too. Pixels with no retrieval are already NaN
    # from ingest, so filter_valid_spectra drops them without the flags.
    keep = filter_valid_spectra(pace_spectra_flat, wavelengths=wavelengths,
                                swir_thresh=swir_thresh, pct=None, band_range=band_range)
    valid_indices = np.where(keep)[0]
    # granules do not all share a line count, so take the shape from this scene
    shape = pace_xr[spectra_var].shape[:2]
    if not len(valid_indices):
        return None

    bands = band_mask(wavelengths, band_range)
    binned, _ = bin_bands(pace_spectra_flat[np.ix_(valid_indices, bands)],
                          wavelengths[bands])
    Y = normalize(binned).T                                            # (L, n_valid)
    A = model.transform(Y, endmembers)                                 # (p, n_valid)

    # residual on unit-norm spectra, so ||y|| == 1 and this is already relative
    difference = Y - endmembers @ A
    residuals = np.linalg.norm(difference, axis=0)

    # PACE_OCI.20250630T111438... -> 2025-06-30, matching visualize_owt.scene_label
    filename = os.path.basename(filepath)
    stamp = filename.split('.')[1]
    return {
        'filename': filename,
        'label': f'{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}',
        'valid_indices': valid_indices,
        'abundances': A,
        'residuals': residuals,
        'spectral_residual': difference.mean(axis=1),
        'abundance_map': _scatter(A.T, valid_indices, shape),
        'residual_map': _scatter(residuals, valid_indices, shape),
        # the maps are swath-indexed, so they are only placeable with these
        'latitude': pace_xr['latitude'].values,
        'longitude': pace_xr['longitude'].values,
    }


def matched_file(directory, filename):
    '''The granule in `directory` sharing this AOP granule's timestamp.'''
    if not os.path.isdir(directory):
        return None
    stem = filename.split('.L2.')[0]
    hits = [f for f in os.listdir(directory)
            if f.startswith(stem) and f.endswith('_water.nc')]
    return os.path.join(directory, hits[0]) if hits else None


def _read(ds, variable, wavelength):
    '''
    One variable as a flat (n_pixels,) array, sampled at the nearest band if the
    field is hyperspectral. None if the variable is absent.
    '''
    if variable not in ds.data_vars:
        return None
    values = ds[variable].values
    if values.ndim == 3:
        bands = ds['wavelength_3d'].values
        values = values[:, :, int(np.argmin(np.abs(bands - wavelength)))]
    return values.ravel()


def load_constituents(filename, wavelength=config.constituent_wavelength):
    '''
    Flat (n_pixels,) arrays describing this scene's water constituents.

    Everything below is computed, then narrowed to `config.constituent_columns`.
    The shares are dimensionless fractions of the absorption and backscatter
    budgets; the rest are absolute magnitudes. The water terms come out of NASA's
    own totals by subtraction, so no pure-water constants are assumed.

    Anything whose granule or variable is missing is simply absent from the
    result, so the figures shrink rather than fail.

    See working-notes/design_notes.md, "Which constituents to compare against".
    '''
    out = {}

    iop = matched_file(config.iop_path, filename)
    if iop is not None:
        ds = xr.open_dataset(iop)
        a = _read(ds, 'a', wavelength)              # total absorption, m-1
        bb = _read(ds, 'bb', wavelength)            # total backscatter, m-1
        aph = _read(ds, 'aph', wavelength)          # phytoplankton absorption
        adg = _read(ds, 'adg_442', wavelength)      # CDOM + detritus absorption
        bbp = _read(ds, 'bbp_442', wavelength)      # particulate backscatter

        # shares -- fractions of a budget. Named "share" throughout: they say what
        # a pixel is made of, not how much of it there is.
        with np.errstate(invalid='ignore', divide='ignore'):
            if a is not None and aph is not None:
                out['aph/a (phyto share)'] = aph / a
            if a is not None and adg is not None:
                out['adg/a (CDOM share)'] = adg / a
            if bb is not None and bbp is not None:
                out['bbp/bb (TSS share)'] = bbp / bb
            if a is not None and aph is not None and adg is not None:
                out['aw/a (clear share)'] = (a - aph - adg) / a
            if a is not None and bb is not None:
                out['u = bb/(a+bb)'] = bb / (a + bb)

        # adg_s is deliberately absent: GIOP fixes it at 0.018, so it is constant
        for label, variable in (('bbp_s', 'bbp_s'),
                                ('aph (phyto)', 'aph'), ('adg_442 (CDOM)', 'adg_442'),
                                ('bbp_442 (TSS)', 'bbp_442'), ('Kd', 'Kd')):
            values = _read(ds, variable, wavelength)
            if values is not None:
                out[label] = values

    bgc = matched_file(config.bgc_path, filename)
    if bgc is not None:
        ds = xr.open_dataset(bgc)
        for label, variable in (('chl-a', 'chlor_a'), ('poc', 'poc')):
            values = _read(ds, variable, wavelength)
            if values is not None:
                out[label] = values

    # keep only the selected columns, in the order config lists them
    selected = {label: out[label]
                for label in config.constituent_columns if label in out}

    # a name that matches nothing would otherwise disappear from the figures with
    # no sign of why, looking identical to a missing granule
    unknown = [label for label in config.constituent_columns if label not in out]
    if unknown:
        print(f'[WARN] - constituent_columns not produced for '
              f'{os.path.basename(filename)}: {unknown}')
    return selected


def save_endmembers(path, endmembers, wavelengths, scenes):
    """
    Persist the fit together with the settings it depends on. Endmembers are only
    meaningful on the band grid they were fitted on, so load_endmembers() refuses
    a file whose provenance does not match.
    """
    np.savez(path, endmembers=endmembers, wavelengths=wavelengths,
             n_components=n_components, spectra_var=spectra_var,
             band_range=np.array(band_range, dtype=float),
             binning_factor=binning_factor,
             training_reject_flags=np.array(training_reject_flags),
             scenes=np.array([os.path.basename(f) for f in scenes]))
    print(f"[INFO] - saved {endmembers.shape[1]} endmembers to {path}")


def load_endmembers(path):
    """
    Read back a saved fit, or raise if it does not match the current settings.

    Returns (endmembers (L, p), wavelengths (L,), scenes it was fitted on).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} does not exist -- set recompute_endmembers = True to fit first')

    stored = np.load(path, allow_pickle=True)
    if int(stored['n_components']) != n_components:
        raise ValueError(f"saved fit has {int(stored['n_components'])} endmembers, "
                         f'but n_components is {n_components}')
    if str(stored['spectra_var']) != spectra_var:
        raise ValueError(f"saved fit used {str(stored['spectra_var'])!r}, "
                         f'but spectra_var is {spectra_var!r}')
    if not np.allclose(stored['band_range'], np.array(band_range, dtype=float)):
        raise ValueError(f"saved fit used band_range {tuple(stored['band_range'])}, "
                         f'but band_range is {band_range}')
    # a fit binned differently lives on a different band grid, so transform()
    # would fail on the shape rather than say what actually went wrong
    if int(stored.get('binning_factor', 1)) != binning_factor:
        raise ValueError(f"saved fit used binning_factor "
                         f"{int(stored.get('binning_factor', 1))}, "
                         f'but binning_factor is {binning_factor}')

    scenes = [str(x) for x in stored['scenes']]
    print(f"[INFO] - loaded {int(stored['n_components'])} endmembers from {path} "
          f'(fitted on {len(scenes)} scenes)')
    return stored['endmembers'], stored['wavelengths'], scenes


def plot_analysis(label, results, components, wavelengths, residual_high=None):
    '''
    The archetype spectra, then one figure per scene holding that scene's
    abundance maps plus its reconstruction residual as a final panel.

    The residual panel is what makes the maps readable: a region can look
    confidently assigned to one archetype while being poorly reconstructed by all
    of them, and only the residual shows it.

    Maps are regridded from swath geometry onto the shared config.roi_box grid
    and drawn north-up over coastlines -- see visualize_owt.regrid_to_roi.
    '''
    # three map columns are too narrow for the full names
    names = vo.archetype_short_names(n_components)

    vo.save_figure(vo.plot_endmember_overlay(components, wavelengths),
                   f"{label}_endmember_loadings.png")
    plt.show()

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(alpha=0)
    residual_cmap = plt.get_cmap('magma').copy()
    residual_cmap.set_bad(alpha=0)

    # one extra panel for the residual, so the grid grows by one. Three
    # columns rather than two: the resulting landscape figure fits a poster
    # column at roughly a third of the height a 2-wide grid needs.
    #
    # Every panel is drawn on the same projected ROI extent rather than on the
    # granule's own pixel indices, so panels are comparable within a scene and
    # between scenes -- the swaths differ in shape, footprint and orientation.
    n_panels = n_components + 1
    map_cols = 3
    map_rows = -(-n_panels // map_cols)

    projection = vo.map_projection()

    for result in results:
        # abundances and residual regridded together, so one binning pass places
        # every panel on exactly the same cells and the residual panel lines up
        # with the abundance panels pixel for pixel
        stacked = np.concatenate(
            [result['abundance_map'], result['residual_map'][:, :, None]], axis=2)
        gridded, extent = vo.regrid_to_roi(stacked, result['latitude'],
                                           result['longitude'])

        # constrained layout, because the shared colourbars below span several
        # axes and tight_layout cannot place those
        fig, ax = plt.subplots(map_rows, map_cols,
                               figsize=(2*vo.column_width_in,1.8*
                                        vo.map_row_height_in() * map_rows),
                               squeeze=False, layout='constrained',
                               subplot_kw={'projection': projection})
        abundance_im = None
        for i in range(n_panels):
            axis = ax.flatten()[i]
            # No per-panel graticule labels. Six panels each carrying lon/lat
            # ticks spend more of the figure on repeated numbers than on maps,
            # and labelling just one makes that panel a different size from the
            # rest, which constrained layout resolves by eating its neighbour's
            # title. The ROI is stated once in the suptitle instead, and the
            # coastline is what a reader actually navigates by.
            vo.setup_map_axis(axis)
            # the raster is plate carree (regular lon/lat) even though the axes
            # are conic -- imshow's transform is what reprojects it
            common = dict(extent=extent, transform=ccrs.PlateCarree(),
                          origin='upper', zorder=2, interpolation='nearest')
            if i < n_components:
                abundance_im = axis.imshow(
                    np.ma.masked_invalid(gridded[:, :, i]),
                    cmap=cmap, vmin=0, vmax=1, **common)
                axis.set_title(names[i])
            else:
                residual_im = axis.imshow(np.ma.masked_invalid(gridded[:, :, -1]),
                                          cmap=residual_cmap, vmin=0,
                                          vmax=residual_high, **common)
                axis.set_title(f"residual")
                # axis.set_title(f"residual  median "
                #                f"{np.median(result['residuals']):.3f}")
                # horizontal, under its own panel: a second vertical bar landed
                # in the same column as the shared one below and overlapped it
                fig.colorbar(residual_im, ax=axis, location='bottom',
                             fraction=0.05, pad=0.02, label='residual')
        for i in range(n_panels, map_rows * map_cols):
            ax.flatten()[i].axis('off')

        # one bar for all five abundance panels: they share the 0-1 scale, so
        # five identical colourbars were spending a quarter of the figure
        # restating it
        if abundance_im is not None:
            fig.colorbar(abundance_im,
                         ax=[ax.flatten()[i] for i in range(n_components)],
                         fraction=0.046, shrink=0.6, label='abundance')

        extent_label = ''
        if config.roi_box is not None:
            lon0, lon1, lat0, lat1 = config.roi_box
            extent_label = f"   {lat0:.0f}-{lat1:.0f}N, {lon0:.0f}-{lon1:.0f}E"
        fig.suptitle(f"{label} - {result['label']}{extent_label}")
        vo.save_figure(fig, f"abundances/{label}_{result['label']}_abundance_maps.png")
        plt.show()

# %% load data

all_pace_spectra_flat = None
wavelengths = None
processed_filepaths = []  # scenes kept, reused later to unmix every water pixel

for filename in sorted(f for f in os.listdir(data_path) if f.endswith('_water.nc')):
    filepath = os.path.join(data_path, filename)
    print(f"Processing {filepath}...")

    pace_xr = xr.open_dataset(filepath)

    scene_wavelengths = pace_xr['wavelength_3d'].values
    pace_spectra_flat = pace_xr[spectra_var].values.reshape(-1, len(scene_wavelengths))
    usable = filter_valid_spectra(pace_spectra_flat, wavelengths=scene_wavelengths,
                                  swir_thresh=swir_thresh, band_range=band_range)

    # training sees only clean water; the quality-flagged pixels are unmixed later
    keep = usable & flag_reject_mask(pace_xr, training_reject_flags)

    # High-latitude winter scenes can come back empty -- above ~55 deg the
    # December sun never clears the HISOLZEN threshold.
    if keep.sum() < min_usable_spectra:
        print(f"  skipped: only {keep.sum()} training spectra "
              f"(need {min_usable_spectra})")
        continue

    processed_filepaths.append(filepath)
    bands = band_mask(scene_wavelengths, band_range)
    if wavelengths is None:
        _, wavelengths = bin_bands(scene_wavelengths[bands], scene_wavelengths[bands])

    if recompute_endmembers:
        pace_nan_removed, _ = bin_bands(pace_spectra_flat[keep][:, bands],
                                        scene_wavelengths[bands])

        # a spectrally diverse subset rather than a uniform random one
        selected = select_spectra(normalize(pace_nan_removed), num_pixels_per_scene,
                                  seed=selection_seed)
        print(f"  {keep.sum():,} training spectra of {usable.sum():,} usable "
              f"({usable.sum() - keep.sum():,} quality-flagged, mapped but not fitted) "
              f"-> selected {len(selected)}")

        chosen = pace_nan_removed[selected, :]
        all_pace_spectra_flat = (chosen if all_pace_spectra_flat is None
                                 else np.concatenate((all_pace_spectra_flat, chosen), axis=0))
    else:
        print(f"  {keep.sum():,} training spectra (not fitting -- "
              f"endmembers will be loaded)")

    if len(processed_filepaths) >= n_scenes:
        break


# %% run model

# needed either way: transform() applies fixed endmembers to new pixels, which is
# what turns a loaded fit into abundance maps
model = BlindEDAA(**config.edaa_kwargs)

if recompute_endmembers:
    data_cube_norm = normalize(np.array(all_pace_spectra_flat))
    endmembers, _ = model.solve(Y=data_cube_norm.T, p=n_components)
    save_endmembers(endmember_path, endmembers, wavelengths, processed_filepaths)
else:
    endmembers, wavelengths, fitted_on = load_endmembers(endmember_path)
    missing = set(fitted_on) - {os.path.basename(f) for f in processed_filepaths}
    if missing:
        print(f'[WARN] - saved fit used {len(missing)} scene(s) not being mapped now: '
              f'{sorted(missing)}')

print("unmixing every water pixel in each processed scene...")
results = [r for r in (unmix_scene(filepath, model, endmembers)
                       for filepath in processed_filepaths) if r is not None]
names = vo.archetype_names(n_components)

# %% interpret the archetypes

for result in results:
    result['constituents'] = load_constituents(result['filename'])

shapes = ie.shape_report(endmembers, wavelengths)
rho, present, counts = ie.correlate_archetypes(results)
labels = ie.label_archetypes(rho, present, shapes)
# ie.print_label_table(shapes, rho, present, labels, n_scenes=len(results))


# %% plot

# fonts at their printed size, for figures placed on the A0 poster
vo.apply_poster_style()


def show(fig, name):
    '''Display a figure, and save it when `save_figures` is set.'''
    if fig is None:
        return
    if save_figures:
        vo.save_figure(fig, name)
    plt.show()


# the endmember loadings and per-scene abundance maps
# plot_analysis('EDAA', results, endmembers.T, wavelengths,
            #   residual_high=vo.residual_scale(results))

# 1. what each archetype is made of
show(vo.plot_constituent_heatmap(rho, present, counts, names),
     '1a_constituent_correlation_heatmap.png')
show(vo.plot_constituent_violins(results, names), '1b_constituent_violin_plots.png')

# 2. where the model fails
show(vo.plot_spectral_residual(results, wavelengths), '2c_spectral_residual.png')

# 3. how composition varies between scenes
# show(plot_scene_small_multiples(results, names),
    #  '3c_scene_composition_small_multiples.png')
# show(vo.plot_scene_small_multiples(results, names),
    #  '3c_scene_composition_small_multiples.png')


# %% summary

# The numbers a talk or a poster caption actually quotes, printed once so they
# are not read off a figure by eye.
pooled_residuals = np.concatenate([r['residuals'] for r in results])
print(f'\n{len(results)} scenes, {pooled_residuals.size:,} unmixed pixels')
print(f'{"scene":>12}{"pixels":>12}{"median":>10}{"p90":>10}')
for result in results:
    print(f'{result["label"]:>12}{result["residuals"].size:>12,}'
          f'{np.median(result["residuals"]):>10.4f}'
          f'{np.percentile(result["residuals"], 90):>10.4f}')
print(f'{"ALL":>12}{pooled_residuals.size:>12,}'
      f'{np.median(pooled_residuals):>10.4f}'
      f'{np.percentile(pooled_residuals, 90):>10.4f}')
