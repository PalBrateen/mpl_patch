"""
mpl_figure_saver.py — MATLAB-like .fig saving for matplotlib (v2)
==================================================================

Patches the matplotlib "Save" button across ALL backends:

  • Qt (PySide6/PyQt) toolbar   → custom dialog with checkboxes
  • ipympl widget (%matplotlib widget / VSCode)  → auto-saves to disk alongside download
  • Programmatic fig.savefig()  → auto-creates .mplpkl alongside

Each save creates a folder:

    my_plot/
      ├── my_plot.png              ← normal image (always)
      ├── my_plot_data.npz         ← x/y arrays extracted from all axes (optional)
      └── my_plot.mplpkl           ← pickle bundle for quick reopen (optional)

Usage — Install once at the top of your script or notebook:
    import mpl_figure_saver
    mpl_figure_saver.install()

    # For widget backend, optionally set where disk saves go:
    mpl_figure_saver.set_save_dir('./figures')

    # Control what gets saved:
    mpl_figure_saver.configure(save_data=True, save_pickle=False)

Usage — Manual save / load:
    mpl_figure_saver.savefig_bundle(fig, 'my_plot.mplpkl')
    fig, extra = mpl_figure_saver.loadfig('my_plot.mplpkl')

    mpl_figure_saver.save_plot_data(fig, 'my_plot_data.npz')
    data = mpl_figure_saver.load_plot_data('my_plot_data.npz')

Usage — ExperimentResult (bundles raw data + analysis + figure):
    from mpl_figure_saver import ExperimentResult
    result = ExperimentResult('ESR #42', params={...})
    result.store('frequencies', freq_array)
    result.attach_figure(fig)
    result.save('esr_042.expr')

    loaded = ExperimentResult.load('esr_042.expr')
    loaded['frequencies']        # raw data back
    loaded.get_figure()          # editable figure back

Author : Claude / BP Lab
Date   : 2026-04
License: MIT
"""

from __future__ import annotations

import pickle
import gzip
import warnings
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Lazy imports — don't force backend selection at import time
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
        _mpl = mpl
        _plt = plt
        _Figure = Figure


# ============================================================================
#  DATA EXTRACTION — pull x/y/image arrays from a matplotlib figure
# ============================================================================

def extract_plot_data(fig) -> dict:
    """
    Walk every Axes in `fig` and extract the underlying numeric data.

    Handles: Line2D (plot), PathCollection (scatter), AxesImage (imshow),
    ErrorbarContainer, bar containers, and text annotations.

    Returns
    -------
    dict with structure:
        {
            'axes': [
                {
                    'title': str,
                    'xlabel': str,  'ylabel': str,
                    'xlim': (lo, hi), 'ylim': (lo, hi),
                    'lines': [
                        {'x': ndarray, 'y': ndarray, 'label': str, 'color': ..., 'linestyle': ...},
                        ...
                    ],
                    'scatter': [
                        {'x': ndarray, 'y': ndarray, 'label': str},
                        ...
                    ],
                    'images': [
                        {'data': ndarray, 'extent': ..., 'cmap': str},
                        ...
                    ],
                    'errorbars': [
                        {'x': ndarray, 'y': ndarray, 'yerr': ndarray, 'xerr': ndarray, 'label': str},
                        ...
                    ],
                    'bars': [
                        {'x': ndarray, 'heights': ndarray, 'widths': ndarray, 'label': str},
                        ...
                    ],
                },
                ...
            ],
            'suptitle': str,
        }
    """
    _ensure_mpl()
    from matplotlib.collections import PathCollection
    from matplotlib.container import ErrorbarContainer, BarContainer
    from matplotlib.image import AxesImage

    result = {
        'axes': [],
        'suptitle': fig._suptitle.get_text() if fig._suptitle else '',
    }

    for ax in fig.axes:
        ax_data = {
            'title': ax.get_title(),
            'xlabel': ax.get_xlabel(),
            'ylabel': ax.get_ylabel(),
            'xlim': tuple(float(v) for v in ax.get_xlim()),
            'ylim': tuple(float(v) for v in ax.get_ylim()),
            'lines': [],
            'scatter': [],
            'images': [],
            'errorbars': [],
            'bars': [],
        }

        # --- Errorbar containers (extract BEFORE lines to tag errorbar lines) ---
        errorbar_line_ids = set()
        if hasattr(ax, 'containers'):
            for cont in ax.containers:
                if isinstance(cont, ErrorbarContainer):
                    data_line = cont[0]  # the main Line2D
                    if data_line is not None:
                        errorbar_line_ids.add(id(data_line))
                        xd = np.asarray(data_line.get_xdata(), dtype=np.float64)
                        yd = np.asarray(data_line.get_ydata(), dtype=np.float64)

                        # Extract error magnitudes from the cap/bar lines
                        yerr = _extract_errorbar_err(cont, axis='y')
                        xerr = _extract_errorbar_err(cont, axis='x')

                        ax_data['errorbars'].append({
                            'x': xd, 'y': yd,
                            'yerr': yerr, 'xerr': xerr,
                            'label': data_line.get_label(),
                        })

                elif isinstance(cont, BarContainer):
                    patches = cont.patches
                    if patches:
                        xs = np.array([p.get_x() for p in patches])
                        heights = np.array([p.get_height() for p in patches])
                        widths = np.array([p.get_width() for p in patches])
                        ax_data['bars'].append({
                            'x': xs, 'heights': heights, 'widths': widths,
                            'label': cont.get_label(),
                        })

        # --- Line2D objects (skip errorbar internals) ---
        for line in ax.get_lines():
            if id(line) in errorbar_line_ids:
                continue
            label = line.get_label()
            if label.startswith('_'):  # matplotlib internal lines
                continue
            xd = np.asarray(line.get_xdata(), dtype=np.float64)
            yd = np.asarray(line.get_ydata(), dtype=np.float64)
            ax_data['lines'].append({
                'x': xd, 'y': yd,
                'label': label,
                'color': line.get_color(),
                'linestyle': line.get_linestyle(),
                'linewidth': line.get_linewidth(),
                'marker': line.get_marker(),
            })

        # --- Scatter (PathCollection) ---
        for coll in ax.collections:
            if isinstance(coll, PathCollection):
                offsets = np.asarray(coll.get_offsets(), dtype=np.float64)
                if offsets.size == 0:
                    continue
                entry = {
                    'x': offsets[:, 0],
                    'y': offsets[:, 1],
                    'label': coll.get_label(),
                }
                # Color array if present
                fc = coll.get_facecolors()
                if fc is not None and len(fc) > 1:
                    entry['facecolors'] = np.asarray(fc)
                # Sizes
                sizes = coll.get_sizes()
                if sizes is not None and len(sizes) > 1:
                    entry['sizes'] = np.asarray(sizes)
                ax_data['scatter'].append(entry)

        # --- Images (imshow) ---
        for img_obj in ax.get_images():
            arr = np.asarray(img_obj.get_array())
            ax_data['images'].append({
                'data': arr,
                'extent': img_obj.get_extent(),
                'cmap': img_obj.get_cmap().name if img_obj.get_cmap() else None,
                'clim': img_obj.get_clim(),
            })

        result['axes'].append(ax_data)

    return result


def _extract_errorbar_err(container, axis='y'):
    """Extract error magnitudes from an ErrorbarContainer."""
    # container = (data_line, cap_lines, bar_line_collections)
    barlines = container[2]  # list of LineCollections
    if not barlines:
        return None

    data_line = container[0]
    data_vals = np.asarray(
        data_line.get_ydata() if axis == 'y' else data_line.get_xdata(),
        dtype=np.float64
    )

    idx = 0 if axis == 'x' else (0 if len(barlines) == 1 else
                                  (1 if len(barlines) == 2 else 0))
    # For y errors: barlines[0] if only yerr, barlines[-1] if both xerr+yerr
    try:
        if axis == 'y':
            bc = barlines[-1] if len(barlines) >= 1 else None
        else:
            bc = barlines[0] if len(barlines) >= 2 else None

        if bc is None:
            return None

        segments = bc.get_segments()
        if not segments:
            return None

        # Each segment is [[x, y_lo], [x, y_hi]]
        err_lo = np.array([seg[0][1 if axis == 'y' else 0] for seg in segments])
        err_hi = np.array([seg[1][1 if axis == 'y' else 0] for seg in segments])
        return np.column_stack([data_vals - err_lo, err_hi - data_vals])
    except Exception:
        return None


def save_plot_data(fig, path: Union[str, Path]):
    """
    Extract all numeric data from `fig` and save as a `.npz` file.

    The .npz contains a dict-of-arrays structure.  Load with:
        data = np.load('my_data.npz', allow_pickle=True)
        plot_info = data['plot_data'].item()  # dict
    """
    path = Path(path).with_suffix('.npz')
    plot_data = extract_plot_data(fig)
    np.savez_compressed(str(path), plot_data=plot_data)
    return path


def load_plot_data(path: Union[str, Path]) -> dict:
    """Load a .npz file saved by save_plot_data()."""
    data = np.load(str(path), allow_pickle=True)
    return data['plot_data'].item()


# ============================================================================
#  PICKLE BUNDLE — save / load full matplotlib figure objects
# ============================================================================

_MAGIC = b'MPLPKL01'
_COMPRESSION_LEVEL = 6


def savefig_bundle(fig, path: Union[str, Path], *, compress: bool = True,
                   extra_data: Optional[dict] = None):
    """
    Save a matplotlib Figure to a re-openable `.mplpkl` pickle bundle.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str or Path
    compress : bool
        gzip the pickle (default True, ~3-5× smaller).
    extra_data : dict, optional
        Extra arrays / metadata to store alongside the figure.
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

    payload = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
    if compress:
        payload = _MAGIC + gzip.compress(payload, compresslevel=_COMPRESSION_LEVEL)
    else:
        payload = _MAGIC + payload

    path.write_bytes(payload)
    return path


def loadfig(path: Union[str, Path], *, show: bool = False):
    """
    Load a `.mplpkl` bundle → (fig, extra_data).

    The figure is re-attached to a live canvas so plt.show() works.
    """
    _ensure_mpl()
    path = Path(path)
    raw = path.read_bytes()

    if not raw.startswith(_MAGIC):
        raise ValueError(f"{path} is not a valid .mplpkl file")

    payload = raw[len(_MAGIC):]
    try:
        payload = gzip.decompress(payload)
    except gzip.BadGzipFile:
        pass

    bundle = pickle.loads(payload)
    fig = bundle['figure']

    # Re-attach to a live canvas
    try:
        dummy_manager = _plt.figure()
        _plt.close(dummy_manager)
        fig.set_canvas(dummy_manager.canvas)
        fig.canvas.manager = dummy_manager.canvas.manager
        fig.canvas.manager.canvas.figure = fig
        fig.canvas.manager.set_window_title(
            f"[Loaded] {path.stem}  ({bundle.get('timestamp', '?')})"
        )
    except Exception:
        pass  # headless backend

    if show:
        fig.show()

    return fig, bundle.get('extra_data', {})


# ============================================================================
#  FOLDER-BASED SAVE — image + npz + mplpkl in one directory
# ============================================================================

def save_figure_folder(fig, filepath: Union[str, Path], *,
                       save_data: bool = True,
                       save_pickle: bool = True,
                       savefig_kwargs: Optional[dict] = None):
    """
    Save a figure into a folder containing image + data + pickle.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filepath : path to the image file (e.g. 'plots/esr.png')
        A folder is created alongside: plots/esr/esr.png + esr_data.npz + esr.mplpkl
    save_data : bool
        Save extracted plot data as .npz (default True).
    save_pickle : bool
        Save pickle bundle as .mplpkl (default True).
    savefig_kwargs : dict
        Extra kwargs passed to the real savefig (dpi, bbox_inches, etc.).
    """
    _ensure_mpl()
    filepath = Path(filepath)
    stem = filepath.stem
    ext = filepath.suffix or '.png'
    folder = filepath.parent / stem

    folder.mkdir(parents=True, exist_ok=True)

    img_path = folder / f"{stem}{ext}"
    kwargs = savefig_kwargs or {}

    # Save the normal image — use the ORIGINAL savefig to avoid recursion
    if _original_savefig is not None:
        _original_savefig(fig, str(img_path), **kwargs)
    else:
        fig.savefig(str(img_path), **kwargs)

    saved = [str(img_path)]

    if save_data:
        try:
            npz_path = save_plot_data(fig, folder / f"{stem}_data")
            saved.append(str(npz_path))
        except Exception as e:
            warnings.warn(f"mpl_figure_saver: data extraction failed: {e}")

    if save_pickle:
        try:
            pkl_path = savefig_bundle(fig, folder / stem)
            saved.append(str(pkl_path))
        except Exception as e:
            warnings.warn(f"mpl_figure_saver: pickle save failed: {e}")

    return folder, saved


# ============================================================================
#  TOOLBAR PATCH — custom save dialog with checkboxes
# ============================================================================

_original_savefig = None
_original_toolbar_save = None
_original_widget_save = None
_installed = False

# Persistent checkbox state across save dialogs within one session
_prefs = {
    'save_data': True,
    'save_pickle': True,
}

# Widget backend save directory — where .npz/.mplpkl go for %matplotlib widget
_widget_save_dir: Optional[Path] = None


def _make_patched_toolbar_save(OrigToolbarClass):
    """
    Create a patched save_figure method for a Qt NavigationToolbar.

    Adds checkboxes to the file dialog:
        ☑ Also save plot data (.npz)
        ☑ Also save editable figure (.mplpkl)
    """

    def patched_save_figure(self, *args):
        from matplotlib.backends.qt_compat import QtWidgets, QtCore

        # --- Build the file filter list (same as original) ---
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        default_filetype = self.canvas.get_default_filetype()

        startpath = os.path.expanduser(_mpl.rcParams['savefig.directory'])
        start = os.path.join(startpath, self.canvas.get_default_filename())

        filters = []
        selectedFilter = None
        for name, exts in sorted_filetypes:
            exts_list = " ".join([f'*.{ext}' for ext in exts])
            filt = f'{name} ({exts_list})'
            if default_filetype in exts:
                selectedFilter = filt
            filters.append(filt)
        filters = ';;'.join(filters)

        # --- Create custom file dialog with checkboxes ---
        dlg = QtWidgets.QFileDialog(
            self.canvas.parent(), "Choose a filename to save to"
        )
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setNameFilter(filters)
        if selectedFilter:
            dlg.selectNameFilter(selectedFilter)
        dlg.selectFile(start)

        # Inject checkboxes into the dialog grid layout
        layout = dlg.layout()
        if layout is not None:
            container = QtWidgets.QWidget()
            hbox = QtWidgets.QHBoxLayout(container)
            hbox.setContentsMargins(0, 4, 0, 0)

            cb_data = QtWidgets.QCheckBox("Also save plot data (.npz)")
            cb_data.setChecked(_prefs['save_data'])
            cb_data.setToolTip(
                "Extracts x/y arrays from all lines, scatter, images, and errorbars\n"
                "into a version-proof .npz file you can always reload and replot."
            )

            cb_pickle = QtWidgets.QCheckBox("Also save editable figure (.mplpkl)")
            cb_pickle.setChecked(_prefs['save_pickle'])
            cb_pickle.setToolTip(
                "Saves the full matplotlib Figure object as a pickle.\n"
                "Reopen with mpl_figure_saver.loadfig() to edit titles, labels, etc.\n"
                "Note: may not survive major matplotlib version changes."
            )

            hbox.addWidget(cb_data)
            hbox.addSpacing(20)
            hbox.addWidget(cb_pickle)
            hbox.addStretch()

            row = layout.rowCount()
            layout.addWidget(container, row, 0, 1, layout.columnCount())

        # --- Show dialog ---
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None

        fnames = dlg.selectedFiles()
        if not fnames:
            return None
        fname = fnames[0]

        # Persist checkbox state for the session
        want_data = cb_data.isChecked() if layout else _prefs['save_data']
        want_pickle = cb_pickle.isChecked() if layout else _prefs['save_pickle']
        _prefs['save_data'] = want_data
        _prefs['save_pickle'] = want_pickle

        # Save directory for next time
        if startpath != "":
            _mpl.rcParams['savefig.directory'] = os.path.dirname(fname)

        # --- Perform the saves ---
        try:
            if want_data or want_pickle:
                # Folder-based save: image + extras
                save_figure_folder(
                    self.canvas.figure, fname,
                    save_data=want_data,
                    save_pickle=want_pickle,
                )
            else:
                # Plain image only — original behavior
                self.canvas.figure.savefig(fname)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error saving file", str(e),
                QtWidgets.QMessageBox.StandardButton.Ok,
                QtWidgets.QMessageBox.StandardButton.NoButton,
            )

        return fname

    return patched_save_figure


def _patched_savefig(self, fname, *args, **kwargs):
    """Drop-in replacement for Figure.savefig — also writes .mplpkl alongside."""
    import io as _io
    result = _original_savefig(self, fname, *args, **kwargs)

    # Skip if fname is a BytesIO (widget backend) — the widget patch handles that
    if isinstance(fname, _io.IOBase):
        return result

    try:
        p = Path(str(fname))
        # Don't double-save if this was triggered from the toolbar folder save
        if p.suffix != '.mplpkl':
            savefig_bundle(self, p.with_suffix('.mplpkl'))
    except Exception as e:
        warnings.warn(f"mpl_figure_saver: auto-bundle failed: {e}")

    return result


# ============================================================================
#  WIDGET BACKEND PATCH — %matplotlib widget (ipympl / VSCode / Jupyter)
# ============================================================================

def set_save_dir(path: Union[str, Path]):
    """
    Set the directory where widget saves write .npz / .mplpkl files.

    Call this once in your notebook:
        mpl_figure_saver.set_save_dir('./figures')

    Without this, widget saves go to the current working directory.
    """
    global _widget_save_dir
    _widget_save_dir = Path(path)
    _widget_save_dir.mkdir(parents=True, exist_ok=True)


def configure(*, save_data: Optional[bool] = None, save_pickle: Optional[bool] = None):
    """
    Set which extras are saved by default.

    Parameters
    ----------
    save_data : bool, optional
        Save extracted .npz data (default True).
    save_pickle : bool, optional
        Save .mplpkl pickle bundle (default True).

    Example
    -------
    >>> mpl_figure_saver.configure(save_data=True, save_pickle=False)
    """
    if save_data is not None:
        _prefs['save_data'] = save_data
    if save_pickle is not None:
        _prefs['save_pickle'] = save_pickle


def _get_widget_save_path(fig) -> Path:
    """Determine the save path for a widget figure."""
    base_dir = _widget_save_dir or Path('.')

    # Derive a filename — priority: suptitle > custom window title > figure_N
    name = ''

    # 1. Suptitle (most likely to be a meaningful user-set name)
    if fig._suptitle:
        text = fig._suptitle.get_text().strip()
        if text:
            name = text

    # 2. Window title (only if user set a custom one, not the default "Figure N")
    if not name:
        try:
            if fig.canvas.manager:
                title = fig.canvas.manager.get_window_title().strip()
                # Skip matplotlib's default "Figure N" titles
                if title and not title.startswith('Figure '):
                    name = title
        except Exception:
            pass

    # 3. Fallback: figure_N
    if not name:
        name = f"figure_{fig.number}" if hasattr(fig, 'number') else 'figure'

    # Sanitize for filesystem
    removed = '<>:"/\\|?*\0 '
    name = name.translate({ord(c): '_' for c in removed})
    if not name:
        name = 'figure'

    return base_dir / name


def _make_patched_widget_toolbar_save():
    """
    Create a patched save_figure for ipympl's Toolbar.

    This is the universal hook — works for ALL ipympl versions:
      • Old (≤0.9.x): Toolbar.save_figure → send_event('save') → JS download
      • New (0.10.0+): Toolbar.save_figure → canvas._send_save_buffer() → Python savefig

    The patch:
    1. Calls the original save_figure (browser download proceeds as normal)
    2. ALSO writes image + .npz + .mplpkl to disk

    Prints paths to the notebook output cell so the user sees what was saved.
    """

    def patched_save_figure(self, *args):
        import io as _io
        _ensure_mpl()
        from matplotlib import rcParams

        fig = self.canvas.figure

        # --- 1. Do the original browser download (unchanged) ---
        if _original_widget_save is not None:
            _original_widget_save(self, *args)

        # --- 2. Save extras to disk ---
        save_path = _get_widget_save_path(fig)
        folder = save_path
        stem = save_path.name
        fmt = rcParams.get('savefig.format', 'png')

        saved = []

        # Always save the image to disk
        try:
            folder.mkdir(parents=True, exist_ok=True)
            img_path = folder / f"{stem}.{fmt}"
            buf = _io.BytesIO()
            # Use ORIGINAL savefig to avoid triggering our Figure.savefig patch
            if _original_savefig is not None:
                _original_savefig(fig, buf, format=fmt)
            else:
                fig.savefig(buf, format=fmt)
            img_path.write_bytes(buf.getvalue())
            saved.append(str(img_path))
        except Exception as e:
            warnings.warn(f"mpl_figure_saver: image disk save failed: {e}")

        if _prefs.get('save_data', True):
            try:
                npz_path = save_plot_data(fig, folder / f"{stem}_data")
                saved.append(str(npz_path))
            except Exception as e:
                warnings.warn(f"mpl_figure_saver: data extraction failed: {e}")

        if _prefs.get('save_pickle', True):
            try:
                pkl_path = savefig_bundle(fig, folder / stem)
                saved.append(str(pkl_path))
            except Exception as e:
                warnings.warn(f"mpl_figure_saver: pickle save failed: {e}")

        if saved:
            print(f"📁 Saved to: {folder}/")
            for s in saved:
                print(f"   {Path(s).name}")

    return patched_save_figure


def install(*, patch_savefig: bool = True, patch_toolbar: bool = True,
            patch_widget: bool = True):
    """
    Install the save patches.

    Parameters
    ----------
    patch_savefig : bool
        Monkey-patch Figure.savefig() to auto-create .mplpkl (default True).
    patch_toolbar : bool
        Monkey-patch the Qt toolbar save button to show checkboxes (default True).
        Falls back gracefully if Qt is not the active backend.
    patch_widget : bool
        Monkey-patch the ipympl widget save button to also write .npz/.mplpkl
        to disk alongside the browser download (default True).
        Falls back gracefully if ipympl is not installed.
    """
    global _original_savefig, _original_toolbar_save, _original_widget_save, _installed
    _ensure_mpl()

    if _installed:
        return

    # --- Patch Figure.savefig ---
    if patch_savefig:
        _original_savefig = _Figure.savefig
        _Figure.savefig = _patched_savefig

    # --- Patch Qt toolbar save_figure ---
    if patch_toolbar:
        try:
            from matplotlib.backends.backend_qt import NavigationToolbar2QT
            _original_toolbar_save = NavigationToolbar2QT.save_figure
            NavigationToolbar2QT.save_figure = _make_patched_toolbar_save(
                NavigationToolbar2QT
            )
        except ImportError:
            pass  # not using Qt backend — that's fine

    # --- Patch ipympl widget save ---
    if patch_widget:
        try:
            from ipympl.backend_nbagg import Toolbar as IpyToolbar
            _original_widget_save = IpyToolbar.save_figure
            IpyToolbar.save_figure = _make_patched_widget_toolbar_save()
        except ImportError:
            pass  # ipympl not installed — that's fine

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
#  ExperimentResult — bundles raw data + analysis + figure in one file
# ============================================================================

class ExperimentResult:
    """
    Single-file container for experimental data, analysis, and figures.

    Saves as `.expr` — a gzip-compressed pickle with version-proof .npz
    data extraction built in.

    Attributes
    ----------
    name : str
    params : dict          — experiment parameters (sweep ranges, settings)
    data : dict            — named arrays / values (accessed via [])
    figures : dict         — named matplotlib Figure objects
    metadata : dict        — timestamps, versions, notes
    """

    _MAGIC = b'EXPR0002'
    _COMPRESS = 6

    def __init__(self, name: str = '', *,
                 params: Optional[dict] = None,
                 notes: str = ''):
        self.name = name
        self.params = params or {}
        self.data: Dict[str, Any] = {}
        self.figures: Dict[str, Any] = {}
        self.metadata: dict = {
            'created': datetime.now().isoformat(),
            'notes': notes,
        }

    # ---- data storage ---

    def store(self, key: str, value: Any):
        """Store a named piece of data."""
        self.data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    # ---- figures ---

    def attach_figure(self, fig, name: str = 'main'):
        """Attach a matplotlib Figure."""
        self.figures[name] = fig

    def get_figure(self, name: str = 'main', *, show: bool = False):
        """Retrieve an attached figure (re-binds to live canvas)."""
        _ensure_mpl()
        fig = self.figures.get(name)
        if fig is None:
            return None
        try:
            dummy = _plt.figure()
            _plt.close(dummy)
            fig.set_canvas(dummy.canvas)
            fig.canvas.manager = dummy.canvas.manager
            fig.canvas.manager.canvas.figure = fig
            fig.canvas.manager.set_window_title(f"{self.name} — {name}")
        except Exception:
            pass
        if show:
            fig.show()
        return fig

    def get_all_figures(self) -> Dict[str, Any]:
        return {k: self.get_figure(k) for k in self.figures}

    # ---- replot ---

    def replot(self, plot_func: Callable, *args,
               fig_name: str = 'main', **kwargs):
        """
        Re-generate a figure from stored data.

        plot_func(result, *args, **kwargs) → Figure
        """
        fig = plot_func(self, *args, **kwargs)
        self.attach_figure(fig, fig_name)
        return fig

    # ---- persistence ---

    def save(self, path: Union[str, Path]):
        """Save everything to a single `.expr` file."""
        path = Path(path)
        if path.suffix not in ('.expr', '.mplpkl'):
            path = path.with_suffix('.expr')
        path.parent.mkdir(parents=True, exist_ok=True)

        self.metadata['saved'] = datetime.now().isoformat()
        self.metadata['numpy_version'] = np.__version__
        try:
            _ensure_mpl()
            self.metadata['matplotlib_version'] = _mpl.__version__
        except Exception:
            self.metadata['matplotlib_version'] = None

        payload = pickle.dumps(self.__dict__, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = self._MAGIC + gzip.compress(payload, compresslevel=self._COMPRESS)
        path.write_bytes(compressed)

        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"✓ Saved: {path}  ({size_mb:.2f} MB)")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentResult':
        """Load an `.expr` file."""
        path = Path(path)
        raw = path.read_bytes()
        if not raw.startswith(cls._MAGIC):
            raise ValueError(f"{path} is not a valid .expr file")
        payload = raw[len(cls._MAGIC):]
        try:
            payload = gzip.decompress(payload)
        except gzip.BadGzipFile:
            pass
        state = pickle.loads(payload)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj

    def savefig(self, fig=None, path: Union[str, Path] = None, *args, **kwargs):
        """Save figure image + full .expr bundle."""
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

    # ---- display ---

    def __repr__(self):
        n_data = len(self.data)
        n_figs = len(self.figures)
        data_keys = ', '.join(list(self.data.keys())[:5])
        if len(self.data) > 5:
            data_keys += ', …'
        return (f"ExperimentResult('{self.name}', "
                f"data=[{data_keys}] ({n_data} items), "
                f"figures={n_figs})")

    def summary(self):
        """Print a human-readable summary of contents."""
        print(f"╔══ ExperimentResult: {self.name}")
        print(f"║  Created : {self.metadata.get('created', '?')}")
        print(f"║  Saved   : {self.metadata.get('saved', 'not yet')}")
        print(f"║  Notes   : {self.metadata.get('notes', '')}")
        print(f"║")
        if self.params:
            print(f"║  Parameters ({len(self.params)}):")
            for k, v in self.params.items():
                print(f"║    {k}: {v}")
            print(f"║")
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
        print(f"║")
        print(f"║  Figures ({len(self.figures)}): "
              f"{', '.join(self.figures.keys()) or 'none'}")
        print(f"╚{'═' * 50}")


# ============================================================================
#  Universal helpers
# ============================================================================

def save(fig_or_result, path, **kwargs):
    """Universal save — Figure → .mplpkl, ExperimentResult → .expr."""
    if isinstance(fig_or_result, ExperimentResult):
        return fig_or_result.save(path, **kwargs)
    else:
        return savefig_bundle(fig_or_result, path, **kwargs)


def load(path):
    """Universal load — auto-detects .mplpkl vs .expr."""
    path = Path(path)
    raw = path.read_bytes()
    if raw.startswith(ExperimentResult._MAGIC):
        return ExperimentResult.load(path)
    elif raw.startswith(_MAGIC):
        return loadfig(path)
    else:
        raise ValueError(f"Unrecognized file format: {path}")
