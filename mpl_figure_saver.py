"""
mpl_figure_saver.py — MATLAB-like .fig saving for matplotlib (v3)
==================================================================

Patches the matplotlib "Save" button and fig.savefig() so every save also
writes a version-proof data file (.npz) and a quick-reopen pickle (.mplpkl),
and lets you reopen/edit or fully replot later without re-running analysis.

Backends covered
----------------
  • Qt toolbar (PySide6 / PyQt)      → save dialog with checkboxes
  • ipympl widget (%matplotlib widget, VSCode / Jupyter)
        - mode 'dialog'    → local Python save dialog (A); nothing written on cancel
        - mode 'timestamp' → plot_<timestamp>[_suptitle] folder (B); for SSH/headless
  • fig.savefig()                    → folder/flat save alongside the image
  • %matplotlib inline               → only via savefig() (no toolbar exists)

Each save produces (subject to the save_data / save_pickle toggles):

    <name>/
      ├── <name>.<fmt>         ← image in the requested/rc format   (always)
      ├── <name>_data.npz      ← arrays extracted from every axes   (save_data)
      └── <name>.mplpkl        ← pickle bundle for quick reopen      (save_pickle)

Quick start
-----------
    import mpl_figure_saver as mfs

    # One call sets everything. Re-calling it re-applies the config
    # (directory / toggles / widget mode) even though patching runs once.
    mfs.install(
        save_dir='/abs/path/to/figures',   # REQUIRED for the widget 'timestamp' mode
        widget_mode='timestamp',           # 'timestamp' (default) or 'dialog'
        save_data=True, save_pickle=True,
    )

Environment defaults (per machine, so one notebook runs everywhere):
    MPL_FIG_SAVER_DIR   → default save_dir
    MPL_FIG_SAVER_MODE  → default widget_mode ('timestamp' | 'dialog')

Reopen / replot
---------------
    fig, extra = mfs.loadfig('plot_.../plot_....mplpkl')   # pickle → editable figure
    data = mfs.load_plot_data('plot_.../plot_..._data.npz')
    fig  = mfs.replot_from_data(data)                      # rebuild from arrays

ExperimentResult (bundle raw data + analysis + figure in one .expr):
    from mpl_figure_saver import ExperimentResult
    r = ExperimentResult('ESR #42', params={...})
    r.store('frequencies', freq); r.attach_figure(fig)
    r.save('esr_042.expr')
    r = ExperimentResult.load('esr_042.expr')

SECURITY: .mplpkl / .expr use pickle. Only load files you trust — unpickling
runs arbitrary code. The .npz is plain arrays and is always safe to load.

Author : Claude / BP Lab   •   Date: 2026-04 → rev 2026-10   •   License: MIT
"""

from __future__ import annotations

import os
import gzip
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Lazy matplotlib import — don't force a backend at import time
# ---------------------------------------------------------------------------
_mpl = None
_plt = None
_Figure = None


def _ensure_mpl():
    global _mpl, _plt, _Figure
    if _mpl is None:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        _mpl, _plt, _Figure = mpl, plt, Figure


def _loud(msg: str):
    """Make a message hard to miss: stderr banner + a warning.

    Widget toolbar callbacks sometimes route stdout to the kernel log rather
    than the cell, so we use both channels.
    """
    line = "!" * 72
    try:
        import sys
        print(f"\n{line}\n[mpl_figure_saver] {msg}\n{line}", file=sys.stderr, flush=True)
    except Exception:
        pass
    warnings.warn(f"mpl_figure_saver: {msg}", stacklevel=2)


# ============================================================================
#  DATA EXTRACTION
# ============================================================================

def extract_plot_data(fig) -> dict:
    """Walk every Axes in `fig` and pull out the underlying numeric data.

    Captured (round-trippable by replot_from_data): figure size, suptitle; per
    axes the title/labels/limits/scales, a date flag, lines (labeled OR not),
    scatter, errorbars, bars, images, vertical/horizontal reference lines
    (axvline/axhline), standalone text, and whether a legend was shown.

    Also captured: fill_between / fill_betweenx (source arrays reconstructed
    from the polygon, including where= gaps) and twinx/twiny relationships.

    NOT captured as source arrays (pickle still preserves them visually):
    stackplot, pcolormesh / contour (QuadMesh), and annotation arrows.
    """
    _ensure_mpl()
    import matplotlib.dates as mdates
    from matplotlib.collections import PathCollection
    from matplotlib.container import ErrorbarContainer, BarContainer

    result = {
        'format_version': 3,
        'figsize': tuple(float(v) for v in fig.get_size_inches()),
        'suptitle': fig._suptitle.get_text() if fig._suptitle else '',
        'axes': [],
    }

    twin_of, twin_kind = _twin_map(fig)

    for ax_i, ax in enumerate(fig.axes):
        xaxis_tr = ax.get_xaxis_transform()
        yaxis_tr = ax.get_yaxis_transform()

        # Detect date axes so we can store numeric values + a flag.
        def _is_date(axis):
            conv = axis.get_converter() if hasattr(axis, 'get_converter') else None
            return isinstance(conv, mdates.DateConverter) or \
                isinstance(conv, getattr(mdates, '_SwitchableDateConverter', ()))
        x_is_date = _is_date(ax.xaxis)
        y_is_date = _is_date(ax.yaxis)

        def _num(vals, is_date):
            if is_date:
                return np.asarray(mdates.date2num(vals), dtype=np.float64)
            return np.asarray(vals, dtype=np.float64)

        ax_data = {
            'title': ax.get_title(),
            'xlabel': ax.get_xlabel(), 'ylabel': ax.get_ylabel(),
            'xscale': ax.get_xscale(), 'yscale': ax.get_yscale(),
            'xlim': tuple(float(v) for v in ax.get_xlim()),
            'ylim': tuple(float(v) for v in ax.get_ylim()),
            'x_is_date': x_is_date, 'y_is_date': y_is_date,
            'has_legend': ax.get_legend() is not None,
            'twin_of': twin_of[ax_i], 'twin_kind': twin_kind[ax_i],
            'subplot_geom': _subplot_geom(ax),
            'xlabel_color': ax.xaxis.label.get_color(),
            'ylabel_color': ax.yaxis.label.get_color(),
            'xtick_color': _tick_color(ax.xaxis),
            'ytick_color': _tick_color(ax.yaxis),
            'lines': [], 'scatter': [], 'images': [],
            'errorbars': [], 'bars': [], 'vlines': [], 'hlines': [],
            'texts': [], 'fills': [],
        }

        # --- Errorbar / bar containers (first, to tag errorbar lines) ---
        errorbar_line_ids = set()
        for cont in getattr(ax, 'containers', []):
            if isinstance(cont, ErrorbarContainer):
                data_line = cont[0]
                if data_line is not None:
                    errorbar_line_ids.add(id(data_line))
                    ax_data['errorbars'].append({
                        'x': _num(data_line.get_xdata(), x_is_date),
                        'y': _num(data_line.get_ydata(), y_is_date),
                        'yerr': _extract_errorbar_err(cont, 'y'),
                        'xerr': _extract_errorbar_err(cont, 'x'),
                        'label': data_line.get_label(),
                    })
            elif isinstance(cont, BarContainer):
                patches = cont.patches
                if patches:
                    ax_data['bars'].append({
                        'x': np.array([p.get_x() for p in patches]),
                        'heights': np.array([p.get_height() for p in patches]),
                        'widths': np.array([p.get_width() for p in patches]),
                        'label': cont.get_label(),
                    })

        # --- Lines: data lines (kept even if unlabeled) vs axvline/axhline ---
        for line in ax.get_lines():
            if id(line) in errorbar_line_ids:
                continue
            tr = line.get_transform()
            if tr is xaxis_tr:                       # axvline
                xd = np.asarray(line.get_xdata(), dtype=np.float64)
                ax_data['vlines'].append({
                    'x': float(xd[0]) if xd.size else np.nan,
                    'color': line.get_color(), 'linestyle': line.get_linestyle(),
                    'linewidth': line.get_linewidth(), 'label': line.get_label(),
                })
                continue
            if tr is yaxis_tr:                        # axhline
                yd = np.asarray(line.get_ydata(), dtype=np.float64)
                ax_data['hlines'].append({
                    'y': float(yd[0]) if yd.size else np.nan,
                    'color': line.get_color(), 'linestyle': line.get_linestyle(),
                    'linewidth': line.get_linewidth(), 'label': line.get_label(),
                })
                continue
            label = line.get_label()
            ax_data['lines'].append({
                'x': _num(line.get_xdata(), x_is_date),
                'y': _num(line.get_ydata(), y_is_date),
                # auto labels like "_child0" become '' so replot won't legend them
                'label': '' if label.startswith('_') else label,
                'color': line.get_color(), 'linestyle': line.get_linestyle(),
                'linewidth': line.get_linewidth(), 'marker': line.get_marker(),
            })

        # --- Scatter (PathCollection) ---
        for coll in ax.collections:
            if isinstance(coll, PathCollection):
                offsets = np.asarray(coll.get_offsets(), dtype=np.float64)
                if offsets.size == 0:
                    continue
                label = coll.get_label()
                entry = {
                    'x': offsets[:, 0], 'y': offsets[:, 1],
                    'label': '' if label.startswith('_') else label,
                }
                fc = coll.get_facecolors()
                if fc is not None and len(fc) > 1:
                    entry['facecolors'] = np.asarray(fc)
                sizes = coll.get_sizes()
                if sizes is not None and len(sizes) > 1:
                    entry['sizes'] = np.asarray(sizes)
                ax_data['scatter'].append(entry)

        # --- fill_between / fill_betweenx ---
        for coll in ax.collections:
            if getattr(coll, 't_direction', None) is not None:
                fb = _extract_fill(coll)
                if fb is not None:
                    ax_data['fills'].append(fb)

        # --- Images (imshow) ---
        for img_obj in ax.get_images():
            ax_data['images'].append({
                'data': np.asarray(img_obj.get_array()),
                'extent': img_obj.get_extent(),
                'cmap': img_obj.get_cmap().name if img_obj.get_cmap() else None,
                'clim': img_obj.get_clim(),
                'aspect': ax.get_aspect(),
            })

        # --- Standalone text / annotation text ---
        for t in ax.texts:
            s = t.get_text()
            if not s:
                continue
            ax_data['texts'].append({
                'text': s,
                'xy': tuple(float(v) for v in t.get_position()),
                'color': t.get_color(),
            })

        result['axes'].append(ax_data)

    return result


def _extract_fill(coll):
    """Reconstruct fill_between / fill_betweenx source arrays from a
    FillBetweenPolyCollection's polygon path(s).

    Returns a dict {orient, t, lo, hi, facecolor, alpha, label} or None.
    orient 'v' → replot with fill_between(t, lo, hi);
    orient 'h' → replot with fill_betweenx(t, lo, hi).
    Multiple polygons (from where=) are concatenated with NaN gaps.
    """
    _ensure_mpl()
    from matplotlib.path import Path as _Path

    t_dir = getattr(coll, 't_direction', None)   # 'x' vertical, 'y' horizontal
    if t_dir is None:
        return None  # not a fill_between collection (older mpl: not supported)
    orient = 'v' if t_dir == 'x' else 'h'

    t_all, lo_all, hi_all = [], [], []
    for path in coll.get_paths():
        v = path.vertices
        codes = path.codes
        if codes is not None:
            v = v[codes != _Path.CLOSEPOLY]
        if len(v) < 4:
            continue
        n = (len(v) - 2) // 2
        if n < 1:
            continue
        lower = v[1:1 + n]            # (indep, boundary-1)
        upper = v[1 + n + 1:1 + n + 1 + n][::-1]   # reversed (indep, boundary-2)
        if len(upper) != n:
            continue
        if orient == 'v':
            t, lo = lower[:, 0], lower[:, 1]
            hi = upper[:, 1]
        else:                          # horizontal: independent var is y
            t, lo = lower[:, 1], lower[:, 0]
            hi = upper[:, 0]
        if t_all:                      # NaN gap between where-segments
            t_all.append([np.nan]); lo_all.append([np.nan]); hi_all.append([np.nan])
        t_all.append(t); lo_all.append(lo); hi_all.append(hi)

    if not t_all:
        return None
    try:
        fc = coll.get_facecolor()
        facecolor = np.asarray(fc[0]).tolist() if fc is not None and len(fc) else None
    except Exception:
        facecolor = None
    label = coll.get_label()
    return {
        'orient': orient,
        't': np.concatenate(t_all),
        'lo': np.concatenate(lo_all),
        'hi': np.concatenate(hi_all),
        'facecolor': facecolor,
        'alpha': coll.get_alpha(),
        'label': '' if (label or '').startswith('_') else label,
    }


def _subplot_geom(ax):
    """Return (nrows, ncols, start, stop) for a regular-grid axes, else None."""
    try:
        ss = ax.get_subplotspec()
        if ss is None:
            return None
        g = ss.get_geometry()
        return tuple(int(v) for v in g[:4])
    except Exception:
        return None


def _tick_color(axis):
    """Best-effort color of an axis' tick labels (for twinx/twiny fidelity)."""
    try:
        labels = axis.get_majorticklabels()
        if labels:
            return labels[0].get_color()
    except Exception:
        pass
    return None


def _twin_map(fig):
    """Map each axes index → (twin_of_primary_index or None, 'x'|'y'|None).

    A twinx shares the x-axis; a twiny shares the y-axis. The earliest axes in
    a twinned group is treated as the primary.
    """
    axes = fig.axes
    twin_of = [None] * len(axes)
    kind = [None] * len(axes)
    gx = axes[0].get_shared_x_axes() if axes else None
    gy = axes[0].get_shared_y_axes() if axes else None
    for i, ax in enumerate(axes):
        sibs = None
        grp = getattr(ax, '_twinned_axes', None)
        if grp is not None:
            try:
                sibs = [axes.index(s) for s in grp.get_siblings(ax) if s in axes]
            except Exception:
                sibs = None
        if not sibs or len(sibs) < 2:
            continue
        primary = min(sibs)
        if i == primary:
            continue
        p_ax = axes[primary]
        shares_x = gx.joined(ax, p_ax) if gx else False
        shares_y = gy.joined(ax, p_ax) if gy else False
        if shares_x and not shares_y:
            twin_of[i], kind[i] = primary, 'x'
        elif shares_y and not shares_x:
            twin_of[i], kind[i] = primary, 'y'
    return twin_of, kind


def _extract_errorbar_err(container, axis='y'):
    """Extract error magnitudes from an ErrorbarContainer, or None if that
    axis has no error bars (so an xerr-only plot reports yerr=None, not zeros)."""
    barlines = container[2]
    if not barlines:
        return None
    data_line = container[0]
    data_vals = np.asarray(
        data_line.get_ydata() if axis == 'y' else data_line.get_xdata(),
        dtype=np.float64,
    )
    try:
        # Heuristic layout of the bar LineCollections:
        #   yerr only            -> [y]
        #   xerr only            -> [x]
        #   both (xerr, yerr)    -> [x, y]
        if len(barlines) >= 2:
            bc = barlines[1] if axis == 'y' else barlines[0]
        else:
            bc = barlines[0]
        segments = bc.get_segments()
        if not segments:
            return None
        comp = 1 if axis == 'y' else 0
        # A y-error collection has segments that vary in y; if this collection
        # is actually the other axis, its chosen component won't vary -> None.
        lo = np.array([seg[0][comp] for seg in segments], dtype=np.float64)
        hi = np.array([seg[1][comp] for seg in segments], dtype=np.float64)
        if np.allclose(lo, hi):
            return None
        return np.column_stack([data_vals - lo, hi - data_vals])
    except Exception:
        return None


def save_plot_data(fig, path: Union[str, Path]):
    """Extract all numeric data from `fig` → compressed `.npz`.

    Load with load_plot_data(), or:
        d = np.load('x.npz', allow_pickle=True)['plot_data'].item()
    """
    path = Path(path).with_suffix('.npz')
    np.savez_compressed(str(path), plot_data=extract_plot_data(fig))
    return path


def load_plot_data(path: Union[str, Path]) -> dict:
    """Load the full metadata+arrays dict written by save_plot_data().

    Parameters
    ----------
    path : str | Path
        Path to a ``*_data.npz`` file.

    Returns
    -------
    dict
        The complete structure from extract_plot_data() — numeric arrays AND
        style/layout metadata. For just the numbers, use load_arrays().
    """
    data = np.load(str(path), allow_pickle=True)
    return data['plot_data'].item()


def arrays_from_data(data: Union[dict, str, Path]) -> list:
    """Strip an extract/`.npz` dict down to just the numeric arrays per axes.

    Use this when you want the raw data back for further analysis and don't
    care about styling or exact layout (the case where replot's N-row vs grid
    layout is irrelevant). Each list item corresponds to one Axes, in the
    order the axes were stored.

    Parameters
    ----------
    data : dict | str | Path
        An extract_plot_data() dict, or a path to a ``*_data.npz`` file.

    Returns
    -------
    list[dict]
        One dict per axes with keys: ``title, xlabel, ylabel`` and the numeric
        collections ``lines, scatter, errorbars, bars, fills, images, vlines,
        hlines`` — each entry carrying only arrays + its ``label``.

    Examples
    --------
    >>> ax0 = arrays_from_data('run_data.npz')[0]
    >>> x, y = ax0['lines'][0]['x'], ax0['lines'][0]['y']
    """
    if isinstance(data, (str, Path)):
        data = load_plot_data(data)
    out = []
    for ad in data.get('axes', []):
        entry = {k: ad.get(k, '') for k in ('title', 'xlabel', 'ylabel')}
        entry['lines'] = [{'x': l['x'], 'y': l['y'], 'label': l.get('label', '')}
                          for l in ad.get('lines', [])]
        entry['scatter'] = [{'x': s['x'], 'y': s['y'], 'label': s.get('label', '')}
                            for s in ad.get('scatter', [])]
        entry['errorbars'] = [{'x': e['x'], 'y': e['y'], 'xerr': e.get('xerr'),
                               'yerr': e.get('yerr'), 'label': e.get('label', '')}
                              for e in ad.get('errorbars', [])]
        entry['bars'] = [{'x': b['x'], 'heights': b['heights'],
                          'widths': b['widths'], 'label': b.get('label', '')}
                         for b in ad.get('bars', [])]
        entry['fills'] = [{'t': f['t'], 'lo': f['lo'], 'hi': f['hi'],
                           'orient': f['orient'], 'label': f.get('label', '')}
                          for f in ad.get('fills', [])]
        entry['images'] = [{'data': im['data'], 'extent': im.get('extent')}
                           for im in ad.get('images', [])]
        entry['vlines'] = [{'x': v['x']} for v in ad.get('vlines', [])]
        entry['hlines'] = [{'y': h['y']} for h in ad.get('hlines', [])]
        out.append(entry)
    return out


def load_arrays(path: Union[str, Path]) -> list:
    """Convenience: arrays_from_data(load_plot_data(path)).

    Returns just the numeric arrays per axes (see arrays_from_data), leaving
    the full style/layout metadata in load_plot_data() for those who want it.
    """
    return arrays_from_data(load_plot_data(path))


def get_series(data: Union[dict, str, Path]) -> list:
    """Flatten each axes into a uniform list of plottable series with x & y.

    This is the easy way to reuse saved data: every item — line, scatter,
    errorbar, bar, or fill band — comes back with the SAME ``x`` and ``y`` keys,
    so you can loop and plot without knowing its type:

        for s in mfs.get_series('run_data.npz')[0]:   # [0] = first axes
            ax.plot(s['x'], s['y'], label=s['label'])

    Parameters
    ----------
    data : dict | str | Path
        An extract_plot_data() dict, or a path to a ``*_data.npz`` file.

    Returns
    -------
    list[list[dict]]
        One inner list per axes. Each series dict always has:
          ``kind``   : 'line' | 'scatter' | 'errorbar' | 'bar' | 'fill'
                       | 'vline' | 'hline'
          ``label``  : the artist's label ('' if none)
          ``x``, ``y`` : numpy arrays ready to plot (for a 'fill', ``y`` is the
                       lower boundary). For 'vline' ``y`` is None; for 'hline'
                       ``x`` is None.
        Extra keys when relevant:
          'fill'      → ``y_upper`` (upper boundary) and ``orient`` ('v'/'h'),
                        so you can ``ax.fill_between(s['x'], s['y'], s['y_upper'])``
          'errorbar'  → ``xerr``, ``yerr`` (or None)
          'bar'       → ``widths``

    Examples
    --------
    >>> series = mfs.get_series('run_data.npz')
    >>> fig, ax = plt.subplots()
    >>> for s in series[0]:
    ...     if s['kind'] == 'fill':
    ...         ax.fill_between(s['x'], s['y'], s['y_upper'], alpha=0.2,
    ...                         label=s['label'])
    ...     else:
    ...         ax.plot(s['x'], s['y'], label=s['label'])
    """
    axes_arrays = arrays_from_data(data)
    out = []
    for ad in axes_arrays:
        series = []
        for l in ad['lines']:
            series.append({'kind': 'line', 'label': l['label'],
                           'x': l['x'], 'y': l['y']})
        for s in ad['scatter']:
            series.append({'kind': 'scatter', 'label': s['label'],
                           'x': s['x'], 'y': s['y']})
        for e in ad['errorbars']:
            series.append({'kind': 'errorbar', 'label': e['label'],
                           'x': e['x'], 'y': e['y'],
                           'xerr': e.get('xerr'), 'yerr': e.get('yerr')})
        for b in ad['bars']:
            series.append({'kind': 'bar', 'label': b['label'],
                           'x': b['x'], 'y': b['heights'],
                           'widths': b['widths']})
        for f in ad['fills']:
            series.append({'kind': 'fill', 'label': f['label'],
                           'x': f['t'], 'y': f['lo'], 'y_upper': f['hi'],
                           'orient': f['orient']})
        for v in ad['vlines']:
            series.append({'kind': 'vline', 'label': '',
                           'x': np.array([v['x']]), 'y': None})
        for h in ad['hlines']:
            series.append({'kind': 'hline', 'label': '',
                           'x': None, 'y': np.array([h['y']])})
        out.append(series)
    return out


def fix_vscode_widget_style():
    """Make %matplotlib widget figures theme-correctly inside VS Code.

    Injects CSS so the ipywidgets container is transparent (no hard-coded white
    box in dark themes) and widget text/labels follow VS Code's editor colors.
    No-op outside an IPython display context. Call once per notebook, e.g. right
    after mfs.install().
    """
    try:
        from IPython.display import display, HTML
    except Exception:
        return
    display(HTML(
        "<style>"
        ".cell-output-ipywidget-background{background-color:transparent !important;}"
        ".jp-OutputArea-output{background-color:transparent !important;}"
        ":root{"
        "--jp-widgets-color:var(--vscode-editor-foreground);"
        "--jp-widgets-label-color:var(--vscode-editor-foreground);"
        "--jp-widgets-font-size:var(--vscode-editor-font-size);"
        "}"
        ".lm-Widget,.jupyter-widgets{color:var(--vscode-editor-foreground);}"
        "</style>"
    ))


# ============================================================================
#  REPLOT FROM EXTRACTED DATA  (version-proof path — no pickle)
# ============================================================================

def replot_from_data(data: Union[dict, str, Path], *, show: bool = False):
    """Rebuild a Figure from extracted ARRAYS (no pickle) — version-proof.

    Reconstructs lines, scatter, errorbars, bars, images, axvline/axhline,
    text, fill_between/fill_betweenx (incl. where= gaps), scales, labels,
    limits, legend, label/tick colors, figure size, and twinx/twiny axes.
    Regular subplot grids (e.g. 2x2) are rebuilt from stored geometry; an
    irregular gridspec falls back to an N-row column. Does NOT reconstruct
    pcolormesh/contour/stackplot — use loadfig() on the .mplpkl for a
    pixel-faithful copy.

    Parameters
    ----------
    data : dict | str | Path
        An extract_plot_data() dict, or a path to a ``*_data.npz`` file.
    show : bool, default False
        If True, call fig.show() before returning.

    Returns
    -------
    matplotlib.figure.Figure
        A freshly built figure. (For the raw arrays alone, see load_arrays.)
    """
    _ensure_mpl()
    import matplotlib.dates as mdates

    if isinstance(data, (str, Path)):
        data = load_plot_data(data)

    all_axes = data.get('axes', [])
    # Primaries get a cell; twins attach to their primary.
    primaries = [i for i, ad in enumerate(all_axes) if ad.get('twin_of') is None]
    figsize = data.get('figsize') or None

    # If every primary reports the same regular (nrows, ncols) grid with a
    # single-cell span, rebuild that grid (so a 2x2 comes back as a 2x2).
    # Otherwise fall back to an N-row column.
    geoms = [all_axes[i].get('subplot_geom') for i in primaries]
    axmap = {}                                   # data-index → live Axes
    use_grid = (len(primaries) > 0 and all(g is not None for g in geoms)
                and len({(g[0], g[1]) for g in geoms}) == 1
                and all(g[2] == g[3] for g in geoms))
    if use_grid:
        nrows, ncols = geoms[0][0], geoms[0][1]
        fig, axs2d = _plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        flat = axs2d.ravel()
        used = set()
        for i, g in zip(primaries, geoms):
            axmap[i] = flat[g[2]]
            used.add(g[2])
        for k, a in enumerate(flat):             # drop unused empty cells
            if k not in used:
                a.remove()
    else:
        n = max(1, len(primaries))
        fig, axs = _plt.subplots(n, 1, figsize=figsize, squeeze=False)
        axs = axs.ravel()
        for row, i in enumerate(primaries):
            axmap[i] = axs[row]
    for i, ad in enumerate(all_axes):            # create twins on their primary
        if ad.get('twin_of') is not None:
            base = axmap.get(ad['twin_of'])
            if base is None:
                continue
            axmap[i] = base.twinx() if ad.get('twin_kind') == 'x' else base.twiny()

    for i, ad in enumerate(all_axes):
        ax = axmap.get(i)
        if ax is not None:
            _draw_axes(ax, ad, mdates)

    if data.get('suptitle'):
        fig.suptitle(data['suptitle'])
    try:
        fig.tight_layout()
    except Exception:
        pass
    if show:
        fig.show()
    return fig


def _draw_axes(ax, ad, mdates):
    """Draw one extracted axes dict onto a live Axes (used by replot)."""
    for ln in ad.get('lines', []):
        x = mdates.num2date(ln['x']) if ad.get('x_is_date') else ln['x']
        ax.plot(x, ln['y'], label=ln.get('label') or None,
                color=ln.get('color'), linestyle=ln.get('linestyle', '-'),
                linewidth=ln.get('linewidth'), marker=ln.get('marker', ''))
    for sc in ad.get('scatter', []):
        kw = {}
        if 'sizes' in sc:
            kw['s'] = sc['sizes']
        if 'facecolors' in sc:
            kw['c'] = sc['facecolors']
        ax.scatter(sc['x'], sc['y'], label=sc.get('label') or None, **kw)
    for eb in ad.get('errorbars', []):
        yerr = eb.get('yerr'); xerr = eb.get('xerr')
        ax.errorbar(eb['x'], eb['y'],
                    yerr=yerr.T if yerr is not None else None,
                    xerr=xerr.T if xerr is not None else None,
                    fmt='o', label=eb.get('label') or None)
    for b in ad.get('bars', []):
        ax.bar(b['x'], b['heights'], width=b['widths'],
               label=b.get('label') or None, align='edge')
    for fb in ad.get('fills', []):
        kw = {}
        if fb.get('facecolor') is not None:
            kw['color'] = fb['facecolor']        # RGBA carries alpha already
        elif fb.get('alpha') is not None:
            kw['alpha'] = fb['alpha']
        if fb.get('label'):
            kw['label'] = fb['label']
        if fb.get('orient') == 'h':
            ax.fill_betweenx(fb['t'], fb['lo'], fb['hi'], **kw)
        else:
            ax.fill_between(fb['t'], fb['lo'], fb['hi'], **kw)
    for im in ad.get('images', []):
        ax.imshow(im['data'], extent=im.get('extent'),
                  cmap=im.get('cmap'), aspect=im.get('aspect', 'equal'))
        if im.get('clim') is not None:
            ax.images[-1].set_clim(im['clim'])
    for vl in ad.get('vlines', []):
        ax.axvline(vl['x'], color=vl.get('color'),
                   linestyle=vl.get('linestyle', '-'), linewidth=vl.get('linewidth'))
    for hl in ad.get('hlines', []):
        ax.axhline(hl['y'], color=hl.get('color'),
                   linestyle=hl.get('linestyle', '-'), linewidth=hl.get('linewidth'))
    for t in ad.get('texts', []):
        ax.text(t['xy'][0], t['xy'][1], t['text'], color=t.get('color'))

    if ad.get('xscale'):
        ax.set_xscale(ad['xscale'])
    if ad.get('yscale'):
        ax.set_yscale(ad['yscale'])
    ax.set_title(ad.get('title', ''))
    ax.set_xlabel(ad.get('xlabel', ''), color=ad.get('xlabel_color') or 'black')
    ax.set_ylabel(ad.get('ylabel', ''), color=ad.get('ylabel_color') or 'black')
    if ad.get('xtick_color'):
        ax.tick_params(axis='x', labelcolor=ad['xtick_color'])
    if ad.get('ytick_color'):
        ax.tick_params(axis='y', labelcolor=ad['ytick_color'])
    try:
        ax.set_xlim(ad['xlim']); ax.set_ylim(ad['ylim'])
    except Exception:
        pass
    if ad.get('has_legend'):
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()


# ============================================================================
#  PICKLE BUNDLE
# ============================================================================

_MAGIC = b'MPLPKL01'
_COMPRESSION_LEVEL = 6


def savefig_bundle(fig, path: Union[str, Path], *, compress: bool = True,
                   extra_data: Optional[dict] = None):
    """Save a Figure to a re-openable `.mplpkl` pickle bundle.

    If the figure contains an unpicklable artist (e.g. a lambda tick
    formatter), the pickle is skipped with a loud warning rather than
    silently dying — the .npz remains your safety net.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to serialize.
    path : str | Path
        Output path; the ``.mplpkl`` suffix is enforced.
    compress : bool, default True
        gzip the pickle payload (smaller files, negligible load cost).
    extra_data : dict | None
        Arbitrary picklable metadata to store alongside the figure and get
        back as the second return value of loadfig() — e.g.
        ``extra_data={'temperature_K': 4.2, 'popt': fit_params}``. This is the
        ONLY thing that populates loadfig()'s ``extra_data``; nothing is added
        automatically.

    Returns
    -------
    Path | None
        The written path, or None if pickling failed (warning emitted).
    """
    _ensure_mpl()
    path = Path(path).with_suffix('.mplpkl')
    bundle = {
        'figure': fig,
        'matplotlib_version': _mpl.__version__,
        'numpy_version': np.__version__,
        'timestamp': datetime.now().isoformat(),
        'extra_data': extra_data or {},
    }
    try:
        payload = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        _loud(f"pickle failed ({type(e).__name__}: {e}); .mplpkl NOT written for "
              f"{path.name}. The .npz (if enabled) is unaffected. Common cause: "
              f"a lambda in set_major_formatter — use a named function or "
              f"matplotlib.ticker.FormatStrFormatter.")
        return None
    payload = _MAGIC + (gzip.compress(payload, _COMPRESSION_LEVEL) if compress else payload)
    path.write_bytes(payload)
    return path


def loadfig(path: Union[str, Path], *, show: bool = False):
    """Load a `.mplpkl` into a live, editable matplotlib Figure.

    The figure is re-bound to a fresh pyplot-managed canvas so plt.show() and
    interactive editing work. Preserves the ORIGINAL layout exactly (grids,
    gridspec, twins, colors) — unlike replot_from_data(), which rebuilds from
    arrays. Warns if the loading matplotlib differs from the saving version.

    Parameters
    ----------
    path : str | Path
        Path to a ``.mplpkl`` file.
    show : bool, default False
        If True, call fig.show() before returning.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The reconstructed figure.
    extra_data : dict
        Whatever was passed as ``extra_data=`` to savefig_bundle() when the
        file was written — a free-form, user-supplied metadata dict (fit
        parameters, instrument settings, ...). Empty ``{}`` for files produced
        by the automatic toolbar / savefig paths, which don't set it.

    Security
    --------
    Uses pickle — only load files you trust; unpickling runs arbitrary code.
    """
    _ensure_mpl()
    path = Path(path)
    raw = path.read_bytes()
    if not raw.startswith(_MAGIC):
        raise ValueError(f"{path} is not a valid .mplpkl file")

    payload = raw[len(_MAGIC):]
    try:
        payload = gzip.decompress(payload)
    except (gzip.BadGzipFile, OSError):
        pass
    bundle = pickle.loads(payload)
    fig = bundle['figure']

    saved_ver = bundle.get('matplotlib_version')
    if saved_ver and saved_ver != _mpl.__version__:
        warnings.warn(
            f"mpl_figure_saver: .mplpkl was saved with matplotlib {saved_ver}, "
            f"loading with {_mpl.__version__}. If it looks wrong, replot from "
            f"the .npz via replot_from_data().", stacklevel=2)

    # Re-bind to a fresh pyplot-managed canvas so plt.show() finds it.
    # (Create the host figure, point its manager at our figure — do NOT close
    # the host first, which was the v2 bug that orphaned the canvas.)
    try:
        host = _plt.figure()
        manager = host.canvas.manager
        manager.canvas.figure = fig
        fig.set_canvas(manager.canvas)
        try:
            manager.set_window_title(f"[Loaded] {path.stem}")
        except Exception:
            pass
    except Exception:
        pass  # headless (Agg) — figure object is still fully usable

    if show:
        fig.show()
    return fig, bundle.get('extra_data', {})


# ============================================================================
#  FOLDER-BASED SAVE
# ============================================================================

def save_figure_folder(fig, filepath: Union[str, Path], *,
                       save_data: bool = True, save_pickle: bool = True,
                       savefig_kwargs: Optional[dict] = None):
    """Save image + optional .npz + optional .mplpkl into <parent>/<stem>/."""
    _ensure_mpl()
    filepath = Path(filepath)
    stem = filepath.stem
    ext = filepath.suffix or '.png'
    folder = filepath.parent / stem
    folder.mkdir(parents=True, exist_ok=True)

    img_path = folder / f"{stem}{ext}"
    kwargs = savefig_kwargs or {}
    if _original_savefig is not None:
        _original_savefig(fig, str(img_path), **kwargs)
    else:
        fig.savefig(str(img_path), **kwargs)
    saved = [str(img_path)]

    if save_data:
        try:
            saved.append(str(save_plot_data(fig, folder / f"{stem}_data")))
        except Exception as e:
            _loud(f"data extraction failed: {e}")
    if save_pickle:
        p = savefig_bundle(fig, folder / stem)
        if p is not None:
            saved.append(str(p))
    return folder, saved


# ============================================================================
#  STATE
# ============================================================================

_original_savefig = None
_original_toolbar_save = None
_original_widget_save = None
_installed = False

_prefs = {'save_data': True, 'save_pickle': True}

_widget_save_dir: Optional[Path] = None
_widget_mode: str = 'timestamp'          # 'timestamp' (B) | 'dialog' (A)
_widget_browser_download: Optional[bool] = None   # None → auto per mode


# ============================================================================
#  CONFIG (standalone; also reachable through install())
# ============================================================================

def set_save_dir(path: Union[str, Path]):
    """Set the directory for widget disk saves. Resolved to an absolute path
    immediately, so a later os.chdir() cannot move it. Created if missing."""
    global _widget_save_dir
    _widget_save_dir = Path(path).expanduser().resolve()
    _widget_save_dir.mkdir(parents=True, exist_ok=True)
    return _widget_save_dir


def configure(*, save_data: Optional[bool] = None,
              save_pickle: Optional[bool] = None,
              widget_mode: Optional[str] = None,
              widget_browser_download: Optional[bool] = None):
    """Set which extras are saved and how the widget save behaves."""
    global _widget_mode, _widget_browser_download
    if save_data is not None:
        _prefs['save_data'] = save_data
    if save_pickle is not None:
        _prefs['save_pickle'] = save_pickle
    if widget_mode is not None:
        if widget_mode not in ('timestamp', 'dialog'):
            raise ValueError("widget_mode must be 'timestamp' or 'dialog'")
        _widget_mode = widget_mode
    if widget_browser_download is not None:
        _widget_browser_download = widget_browser_download


# ============================================================================
#  WIDGET SAVE PATH
# ============================================================================

def _sanitize(name: str) -> str:
    bad = '<>:"/\\|?*\0 '
    name = name.translate({ord(c): '_' for c in bad}).strip('_')
    return name or 'figure'


def _figure_name_hint(fig) -> str:
    """Best-effort human name: suptitle > custom window title > ''."""
    if fig._suptitle:
        t = fig._suptitle.get_text().strip()
        if t:
            return _sanitize(t)
    try:
        if fig.canvas.manager:
            t = fig.canvas.manager.get_window_title().strip()
            if t and not t.startswith('Figure '):
                return _sanitize(t)
    except Exception:
        pass
    return ''


def _timestamp_folder(fig) -> Path:
    """B: plot_<YYYYMMDD_HHMMSS_ffffff>[_suptitle], collision-checked."""
    base = _widget_save_dir
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    hint = _figure_name_hint(fig)
    stem = f"plot_{stamp}" + (f"_{hint}" if hint else '')
    candidate = base / stem
    i = 1
    while candidate.exists():       # sub-microsecond double-clicks only
        candidate = base / f"{stem}_{i}"
        i += 1
    return candidate


def _write_extras(fig, folder: Path, stem: str, fmt: str, want_data, want_pickle):
    saved = []
    try:
        import io as _io
        folder.mkdir(parents=True, exist_ok=True)
        img_path = folder / f"{stem}.{fmt}"
        buf = _io.BytesIO()
        (_original_savefig or _Figure.savefig)(fig, buf, format=fmt)
        img_path.write_bytes(buf.getvalue())
        saved.append(str(img_path))
    except Exception as e:
        _loud(f"image disk save failed: {e}")
    if want_data:
        try:
            saved.append(str(save_plot_data(fig, folder / f"{stem}_data")))
        except Exception as e:
            _loud(f"data extraction failed: {e}")
    if want_pickle:
        p = savefig_bundle(fig, folder / stem)
        if p is not None:
            saved.append(str(p))
    return saved


def _make_patched_widget_toolbar_save():
    """Patched ipympl Toolbar.save_figure honoring the configured widget mode."""

    def patched_save_figure(self, *args):
        _ensure_mpl()
        from matplotlib import rcParams
        fig = self.canvas.figure

        mode = _widget_mode
        browser = _widget_browser_download
        if browser is None:
            browser = (mode == 'timestamp')   # dialog mode suppresses the dup download

        # ---- mode A: local Python dialog (nothing written on cancel) ----
        if mode == 'dialog':
            choice = _dialog_choose_save(fig, rcParams.get('savefig.format', 'png'))
            if choice is None:
                return  # cancelled → no browser download, no files
            fname, want_data, want_pickle = choice   # fname is the user's path
            if want_data or want_pickle:
                # save_figure_folder nests <parent>/<stem>/ exactly once,
                # the same as the Qt path — pass the raw chosen path straight in.
                folder, saved = save_figure_folder(
                    fig, fname, save_data=want_data, save_pickle=want_pickle)
                _report_saved(folder, saved)
            else:
                p = Path(fname)
                p.parent.mkdir(parents=True, exist_ok=True)
                (_original_savefig or _Figure.savefig)(fig, str(p))
                _report_saved(p.parent, [str(p)])
            return

        # ---- mode B: timestamp folder (SSH / headless) ----
        if browser and _original_widget_save is not None:
            try:
                _original_widget_save(self, *args)   # browser download proceeds
            except Exception as e:
                _loud(f"original widget download failed: {e}")

        if _widget_save_dir is None:
            _loud("widget_mode='timestamp' but no save directory is set. "
                  "Nothing written to disk. Call mfs.install(save_dir=...) or "
                  "mfs.set_save_dir(...) with an absolute path.")
            return

        folder = _timestamp_folder(fig)
        fmt = rcParams.get('savefig.format', 'png')
        saved = _write_extras(fig, folder, folder.name, fmt,
                              _prefs.get('save_data', True),
                              _prefs.get('save_pickle', True))
        _report_saved(folder, saved)

    return patched_save_figure


def _report_saved(folder, saved):
    if saved:
        print(f"📁 Saved to: {folder}")
        for s in saved:
            print(f"   {Path(s).name}")


# ---- Local save dialog for widget mode A (Qt if present, else tkinter) ----

def _dialog_choose_save(fig, default_fmt):
    """Ask the user (via a local tkinter dialog) where to save and what extras.

    Returns ``(chosen_path, want_data, want_pickle)`` or ``None`` if cancelled.
    ``chosen_path`` is passed as-is to save_figure_folder(), which creates the
    single ``<parent>/<stem>/`` folder — so don't pre-nest here.

    Runs on the kernel's machine, so it only makes sense when the kernel has a
    display (local or remote-desktop). On a bare SSH session use the
    'timestamp' widget mode instead. tkinter ships with CPython; if it can't
    open (no display), this warns and returns None.
    """
    want_data = _prefs.get('save_data', True)
    want_pickle = _prefs.get('save_pickle', True)

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        v_data = tk.BooleanVar(value=want_data)
        v_pkl = tk.BooleanVar(value=want_pickle)
        # A tiny options window first, then the file dialog.
        opt = tk.Toplevel(root); opt.title("Save options")
        tk.Checkbutton(opt, text="Also save plot data (.npz)", variable=v_data).pack(anchor='w', padx=10, pady=4)
        tk.Checkbutton(opt, text="Also save editable figure (.mplpkl)", variable=v_pkl).pack(anchor='w', padx=10, pady=4)
        proceed = {'ok': False}
        tk.Button(opt, text="Choose location…",
                  command=lambda: (proceed.update(ok=True), opt.destroy())).pack(pady=6)
        opt.protocol("WM_DELETE_WINDOW", opt.destroy)
        root.wait_window(opt)
        if not proceed['ok']:
            root.destroy(); return None
        initial = (_widget_save_dir or Path.cwd())
        fname = filedialog.asksaveasfilename(
            initialdir=str(initial),
            initialfile=(_figure_name_hint(fig) or 'figure') + f".{default_fmt}",
            defaultextension=f".{default_fmt}",
            filetypes=[("Image", f"*.{default_fmt}"), ("All files", "*.*")])
        root.destroy()
        if not fname:
            return None
        _prefs['save_data'] = v_data.get()
        _prefs['save_pickle'] = v_pkl.get()
        p = Path(fname)
        if not p.suffix:
            p = p.with_suffix(f".{default_fmt}")
        return p, v_data.get(), v_pkl.get()
    except Exception as e:
        _loud(f"dialog mode unavailable ({type(e).__name__}: {e}); no display? "
              f"Use widget_mode='timestamp' on headless/SSH sessions.")
        return None


# ============================================================================
#  Figure.savefig PATCH
# ============================================================================

def _patched_savefig(self, fname, *args, **kwargs):
    """Drop-in Figure.savefig — also writes a .mplpkl (and .npz if enabled)
    alongside. Honors the same save_data toggle as the toolbars."""
    import io as _io
    result = _original_savefig(self, fname, *args, **kwargs)
    if isinstance(fname, _io.IOBase):      # widget BytesIO path handles its own
        return result
    try:
        p = Path(str(fname))
        if p.suffix == '.mplpkl':
            return result
        if _prefs.get('save_pickle', True):
            savefig_bundle(self, p.with_suffix('.mplpkl'))
        if _prefs.get('save_data', True):
            try:
                save_plot_data(self, p.with_suffix('.npz'))
            except Exception as e:
                _loud(f"auto .npz failed: {e}")
    except Exception as e:
        _loud(f"auto-bundle failed: {e}")
    return result


_patched_savefig._mfs_patched = True   # marker so re-install can't re-wrap us


# ============================================================================
#  INSTALL / UNINSTALL
# ============================================================================

def install(*, patch_savefig: bool = True, patch_toolbar: bool = True,
            patch_widget: bool = True,
            save_dir: Optional[Union[str, Path]] = None,
            save_data: Optional[bool] = None, save_pickle: Optional[bool] = None,
            widget_mode: Optional[str] = None,
            widget_browser_download: Optional[bool] = None):
    """Install the save patches and apply configuration in one call.

    The monkey-patching of Figure.savefig and the Qt/ipympl toolbars runs once,
    but the configuration below is (re)applied on EVERY call, so you may call
    install(save_dir=...) again mid-session to re-point the directory or switch
    mode. Safe under %autoreload: re-running never double-wraps the patches.

    Parameters
    ----------
    patch_savefig : bool, default True
        Patch matplotlib.figure.Figure.savefig so it also writes the
        sidecar .mplpkl / .npz next to every image you save in code.
    patch_toolbar : bool, default True
        Patch the Qt NavigationToolbar "Save" button (adds the two checkboxes).
        Silently skipped if no Qt backend is importable.
    patch_widget : bool, default True
        Patch the ipympl (%matplotlib widget) toolbar "Save" button.
        Silently skipped if ipympl is not installed.
    save_dir : str | Path | None
        Directory for widget 'timestamp'-mode saves. Resolved to an absolute
        path immediately (a later os.chdir can't move it) and created if
        missing. REQUIRED for widget_mode='timestamp'; without it those saves
        warn loudly and write nothing. Falls back to $MPL_FIG_SAVER_DIR.
    save_data : bool | None, default True
        Whether every save also writes the version-proof ``*_data.npz``.
        None leaves the current setting unchanged.
    save_pickle : bool | None, default True
        Whether every save also writes the quick-reopen ``.mplpkl``.
        None leaves the current setting unchanged.
    widget_mode : {'timestamp', 'dialog'} | None
        How the ipympl "Save" button behaves:
          'timestamp' — write ``plot_<timestamp>[_suptitle]/`` into save_dir
                         (works headless / over SSH; the browser download, if
                         any, still happens). This is the default.
          'dialog'    — open a local Python save dialog on the kernel's machine
                         (local or remote-desktop only); nothing is written if
                         you cancel. Falls back to $MPL_FIG_SAVER_MODE.
    widget_browser_download : bool | None
        Force the ipympl browser download on/off. None = auto: on for
        'timestamp', off for 'dialog' (so you don't get two dialogs).

    Returns
    -------
    None

    Examples
    --------
    >>> import mpl_figure_saver as mfs
    >>> mfs.install(save_dir='/data/figs', widget_mode='timestamp')   # SSH box
    >>> mfs.install(widget_mode='dialog')                             # laptop
    """
    global _original_savefig, _original_toolbar_save, _original_widget_save, _installed
    _ensure_mpl()

    # --- config first, so a re-call still takes effect past the guard below ---
    env_mode = os.environ.get('MPL_FIG_SAVER_MODE')
    configure(save_data=save_data, save_pickle=save_pickle,
              widget_mode=widget_mode or env_mode,
              widget_browser_download=widget_browser_download)
    env_dir = os.environ.get('MPL_FIG_SAVER_DIR')
    if save_dir is not None:
        set_save_dir(save_dir)
    elif env_dir and _widget_save_dir is None:
        set_save_dir(env_dir)

    if _widget_mode == 'timestamp' and _widget_save_dir is None:
        _loud("widget_mode='timestamp' is set but no save_dir yet. Widget saves "
              "will write nothing until you pass save_dir=... (install) or set "
              "MPL_FIG_SAVER_DIR. Qt and fig.savefig() are unaffected.")

    if _installed:
        return

    # Re-point the class at our (possibly freshly reloaded) patch while
    # recovering the TRUE original. If the live method is already ours — module
    # reloaded via %autoreload, or install() re-run after an edit — naively
    # capturing it would make the patch call itself forever; so we read the real
    # original back from the ._mfs_original attribute we stash on every patch.
    def _true_original(live):
        if getattr(live, '_mfs_patched', False):
            return getattr(live, '_mfs_original', None)
        return live

    if patch_savefig:
        _original_savefig = _true_original(_Figure.savefig)
        _patched_savefig._mfs_original = _original_savefig
        _Figure.savefig = _patched_savefig

    if patch_toolbar:
        try:
            from matplotlib.backends.backend_qt import NavigationToolbar2QT
            _original_toolbar_save = _true_original(NavigationToolbar2QT.save_figure)
            patched = _make_patched_toolbar_save(NavigationToolbar2QT)
            patched._mfs_patched = True
            patched._mfs_original = _original_toolbar_save
            NavigationToolbar2QT.save_figure = patched
        except ImportError:
            pass

    if patch_widget:
        try:
            from ipympl.backend_nbagg import Toolbar as IpyToolbar
            _original_widget_save = _true_original(IpyToolbar.save_figure)
            patched = _make_patched_widget_toolbar_save()
            patched._mfs_patched = True
            patched._mfs_original = _original_widget_save
            IpyToolbar.save_figure = patched
        except ImportError:
            pass

    _installed = True


def uninstall():
    """Remove all monkey-patches."""
    global _original_savefig, _original_toolbar_save, _original_widget_save, _installed
    _ensure_mpl()
    if not _installed:
        return
    if _original_savefig is not None:
        _Figure.savefig = _original_savefig
        _original_savefig = None
    if _original_toolbar_save is not None:
        try:
            from matplotlib.backends.backend_qt import NavigationToolbar2QT
            NavigationToolbar2QT.save_figure = _original_toolbar_save
        except ImportError:
            pass
        _original_toolbar_save = None
    if _original_widget_save is not None:
        try:
            from ipympl.backend_nbagg import Toolbar as IpyToolbar
            IpyToolbar.save_figure = _original_widget_save
        except ImportError:
            pass
        _original_widget_save = None
    _installed = False


# ============================================================================
#  QT TOOLBAR PATCH
# ============================================================================

def _make_patched_toolbar_save(OrigToolbarClass):
    """Qt save_figure with the two extra checkboxes. Fixes the v2 bug where an
    all-unchecked save still wrote a .mplpkl: with neither box ticked, only the
    plain image is written (original behaviour)."""

    def patched_save_figure(self, *args):
        from matplotlib.backends.qt_compat import QtWidgets

        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        default_filetype = self.canvas.get_default_filetype()
        startpath = os.path.expanduser(_mpl.rcParams['savefig.directory'])
        start = os.path.join(startpath, self.canvas.get_default_filename())

        filters, selectedFilter = [], None
        for name, exts in sorted_filetypes:
            exts_list = " ".join(f'*.{e}' for e in exts)
            filt = f'{name} ({exts_list})'
            if default_filetype in exts:
                selectedFilter = filt
            filters.append(filt)
        filters = ';;'.join(filters)

        dlg = QtWidgets.QFileDialog(self.canvas.parent(), "Choose a filename to save to")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setNameFilter(filters)
        if selectedFilter:
            dlg.selectNameFilter(selectedFilter)
        dlg.selectFile(start)

        layout = dlg.layout()
        cb_data = cb_pickle = None
        if layout is not None:
            container = QtWidgets.QWidget()
            hbox = QtWidgets.QHBoxLayout(container)
            hbox.setContentsMargins(0, 4, 0, 0)
            cb_data = QtWidgets.QCheckBox("Also save plot data (.npz)")
            cb_data.setChecked(_prefs['save_data'])
            cb_pickle = QtWidgets.QCheckBox("Also save editable figure (.mplpkl)")
            cb_pickle.setChecked(_prefs['save_pickle'])
            hbox.addWidget(cb_data); hbox.addSpacing(20)
            hbox.addWidget(cb_pickle); hbox.addStretch()
            layout.addWidget(container, layout.rowCount(), 0, 1, layout.columnCount())

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        fnames = dlg.selectedFiles()
        if not fnames:
            return None
        fname = fnames[0]

        want_data = cb_data.isChecked() if cb_data else _prefs['save_data']
        want_pickle = cb_pickle.isChecked() if cb_pickle else _prefs['save_pickle']
        _prefs['save_data'], _prefs['save_pickle'] = want_data, want_pickle

        if startpath != "":
            _mpl.rcParams['savefig.directory'] = os.path.dirname(fname)

        try:
            if want_data or want_pickle:
                save_figure_folder(self.canvas.figure, fname,
                                   save_data=want_data, save_pickle=want_pickle)
            else:
                # neither extra requested → plain image only, via original savefig
                (_original_savefig or _Figure.savefig)(self.canvas.figure, fname)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error saving file", str(e),
                QtWidgets.QMessageBox.StandardButton.Ok,
                QtWidgets.QMessageBox.StandardButton.NoButton)
        return fname

    return patched_save_figure


# ============================================================================
#  ExperimentResult
# ============================================================================

class ExperimentResult:
    """Single-file container for experimental data + analysis + figures (.expr).
    SECURITY: .expr is pickle — only load trusted files."""

    _MAGIC = b'EXPR0002'
    _COMPRESS = 6

    def __init__(self, name: str = '', *, params: Optional[dict] = None, notes: str = ''):
        self.name = name
        self.params = params or {}
        self.data: Dict[str, Any] = {}
        self.figures: Dict[str, Any] = {}
        self.metadata: dict = {'created': datetime.now().isoformat(), 'notes': notes}

    def store(self, key: str, value: Any): self.data[key] = value
    def __getitem__(self, key): return self.data[key]
    def __contains__(self, key): return key in self.data
    def keys(self): return self.data.keys()
    def items(self): return self.data.items()

    def attach_figure(self, fig, name: str = 'main'): self.figures[name] = fig

    def get_figure(self, name: str = 'main', *, show: bool = False):
        _ensure_mpl()
        fig = self.figures.get(name)
        if fig is None:
            return None
        try:
            host = _plt.figure()
            manager = host.canvas.manager
            manager.canvas.figure = fig
            fig.set_canvas(manager.canvas)
            manager.set_window_title(f"{self.name} — {name}")
        except Exception:
            pass
        if show:
            fig.show()
        return fig

    def get_all_figures(self): return {k: self.get_figure(k) for k in self.figures}

    def replot(self, plot_func: Callable, *args, fig_name: str = 'main', **kwargs):
        fig = plot_func(self, *args, **kwargs)
        self.attach_figure(fig, fig_name)
        return fig

    def save(self, path: Union[str, Path]):
        path = Path(path)
        if path.suffix not in ('.expr', '.mplpkl'):
            path = path.with_suffix('.expr')
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata['saved'] = datetime.now().isoformat()
        self.metadata['numpy_version'] = np.__version__
        try:
            _ensure_mpl(); self.metadata['matplotlib_version'] = _mpl.__version__
        except Exception:
            self.metadata['matplotlib_version'] = None
        payload = pickle.dumps(self.__dict__, protocol=pickle.HIGHEST_PROTOCOL)
        path.write_bytes(self._MAGIC + gzip.compress(payload, self._COMPRESS))
        print(f"✓ Saved: {path}  ({path.stat().st_size / 1048576:.2f} MB)")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentResult':
        path = Path(path)
        raw = path.read_bytes()
        if not raw.startswith(cls._MAGIC):
            raise ValueError(f"{path} is not a valid .expr file")
        payload = raw[len(cls._MAGIC):]
        try:
            payload = gzip.decompress(payload)
        except (gzip.BadGzipFile, OSError):
            pass
        obj = cls.__new__(cls)
        obj.__dict__.update(pickle.loads(payload))
        return obj

    def savefig(self, fig=None, path: Union[str, Path] = None, *args, **kwargs):
        _ensure_mpl()
        if fig is None:
            fig = self.figures.get('main')
        if fig is None:
            raise ValueError("No figure to save. Attach one first.")
        if path is not None:
            p = Path(path)
            fig.savefig(str(p), *args, **kwargs)
            self.save(p.with_suffix('.expr'))
        else:
            self.save(Path(self.name.replace(' ', '_')).with_suffix('.expr'))

    def __repr__(self):
        keys = ', '.join(list(self.data.keys())[:5]) + (', …' if len(self.data) > 5 else '')
        return (f"ExperimentResult('{self.name}', data=[{keys}] "
                f"({len(self.data)} items), figures={len(self.figures)})")

    def summary(self):
        print(f"╔══ ExperimentResult: {self.name}")
        print(f"║  Created : {self.metadata.get('created', '?')}")
        print(f"║  Saved   : {self.metadata.get('saved', 'not yet')}")
        print(f"║  Notes   : {self.metadata.get('notes', '')}\n║")
        if self.params:
            print(f"║  Parameters ({len(self.params)}):")
            for k, v in self.params.items():
                print(f"║    {k}: {v}")
            print("║")
        print(f"║  Data ({len(self.data)} items):")
        for k, v in self.data.items():
            if isinstance(v, np.ndarray):
                print(f"║    {k}: ndarray {v.shape} {v.dtype}")
            elif isinstance(v, (list, tuple)):
                print(f"║    {k}: {type(v).__name__} len={len(v)}")
            elif isinstance(v, dict):
                print(f"║    {k}: dict ({len(v)} keys)")
            else:
                print(f"║    {k}: {type(v).__name__} = {v}")
        print(f"║\n║  Figures ({len(self.figures)}): "
              f"{', '.join(self.figures.keys()) or 'none'}")
        print(f"╚{'═' * 50}")


# ============================================================================
#  Universal helpers
# ============================================================================

def save(fig_or_result, path, **kwargs):
    if isinstance(fig_or_result, ExperimentResult):
        return fig_or_result.save(path, **kwargs)
    return savefig_bundle(fig_or_result, path, **kwargs)


def load(path):
    path = Path(path)
    raw = path.read_bytes()
    if raw.startswith(ExperimentResult._MAGIC):
        return ExperimentResult.load(path)
    elif raw.startswith(_MAGIC):
        return loadfig(path)
    raise ValueError(f"Unrecognized file format: {path}")
