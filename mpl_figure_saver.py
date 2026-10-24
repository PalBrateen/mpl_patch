"""
mpl_figure_saver.py — MATLAB-like .fig saving for matplotlib (v4)
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
      ├── <name>.mplscene      ← JSON scene + arrays, NO pickle      (save_data)
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
    fig  = mfs.replot_scene('plot_.../plot_....mplscene')  # pickle-free rebuild
    d    = mfs.read_data('plot_.../plot_....mplscene')     # queryable data view
    x, y = d['signal']                                     # by label, unpacked

ExperimentResult (bundle raw data + analysis + figure in one .expr):
    from mpl_figure_saver import ExperimentResult
    r = ExperimentResult('ESR #42', params={...})
    r.store('frequencies', freq); r.attach_figure(fig)
    r.save('esr_042.expr')
    r = ExperimentResult.load('esr_042.expr')

SECURITY: .mplpkl / .expr use pickle — only load files you trust, since
unpickling runs arbitrary code. The .mplscene file contains NO pickle (JSON +
arrays loaded with allow_pickle=False) and is safe to load from any source.

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

import mpl_scene as _scene

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

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
linestyles = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as ':'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot'),  # Same as '-.'
    #  ('loosely dotted',        (0, (1, 10))),
    #  ('dotted',                (0, (1, 5))),
     ('densely dotted',        (0, (1, 1))),

    #  ('long dash with offset', (5, (10, 3))),
    #  ('loosely dashed',        (0, (5, 10))),
    #  ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

    #  ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

    #  ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
    #  ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    #  ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
    ]

def rgb2hex(r, g, b):
    r, g, b = [max(0, min(x, 255)) for x in (r, g, b)]      # Ensure clamp 0 to 255
    return f"#{r:02x}{g:02x}{b:02x}"

def vscode_tweak():
    from IPython.display import display, HTML
    # 1. Force the widget container to be transparent via CSS injection
    # This targets the specific 'lm-Widget' and 'jupyter-widgets' classes
    # /* Remove the hardcoded white background from the interactive widget container */
    # /* Ensure widget text and labels match VS Code foreground colors */
    display(HTML("<style>.cell-output-ipywidget-background {background-color: transparent !important;}"\
    ":root {--jp-widgets-color: var(--vscode-editor-foreground);"
    "--jp-widgets-font-size: var(--vscode-editor-font-size);}</style>"))


# ============================================================================
#  DATA / SCENE  (delegates to mpl_scene: generic artist walk, no pickle)
# ============================================================================
#
# v3 hand-wrote an extractor per plot type, which meant every unsupported type
# was silent data loss. v4 walks the artist tree generically (mpl_scene) and
# stores a JSON scene + raw arrays in a .mplscene zip. Nothing in that file is
# pickled, so it loads with allow_pickle=False and is readable from any
# language.

SCENE_EXT = ".mplscene"


def save_scene_file(fig, path):
    """Write the version-proof scene file (JSON tree + arrays, no pickle).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str | Path
        Output path; the ``.mplscene`` suffix is enforced.

    Returns
    -------
    Path
    """
    path = Path(path).with_suffix(SCENE_EXT)
    _scene.save_scene(fig, str(path))
    return path


def load_scene_file(path):
    """Load a ``.mplscene`` -> (tree, arrays). Mostly for inspection."""
    return _scene.load_scene(str(path))


def replot_scene(path_or_tree, arrays=None, *, show=False):
    """Rebuild a Figure from a ``.mplscene`` file (version-proof path).

    Unlike loadfig() this uses no pickle: the figure is reconstructed by
    replaying artists and properties. Covers lines, scatter, errorbars, bars,
    fills, images, pcolormesh, contours, twins, grids, legends, annotations.

    Parameters
    ----------
    path_or_tree : str | Path | dict
        A ``.mplscene`` path, or a tree dict (then pass ``arrays`` too).
    show : bool, default False

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_mpl()
    if arrays is None:
        tree, arrays = _scene.load_scene(str(path_or_tree))
    else:
        tree = path_or_tree
    fig = _scene.replay_figure(tree, arrays)
    if show:
        fig.show()
    return fig


def read_data(path):
    """Queryable data view of a saved figure — the thing MATLAB .fig lacks.

    Parameters
    ----------
    path : str | Path
        A ``.mplscene`` file.

    Returns
    -------
    mpl_scene.FigureData
        Index by label (``d["signal"]``), unpack to arrays (``x, y = d["signal"]``),
        filter by kind (``d.of_kind("fill")``), export (``d.to_csv(...)``,
        ``d.to_dataframe()``), or re-draw one series (``s.plot(ax)``).

    Examples
    --------
    >>> d = mfs.read_data("run.mplscene")
    >>> d.labels
    ['signal', 'band']
    >>> x, y = d["signal"]
    >>> d["band"].hi[:3]
    """
    return _scene.read_data(str(path))


def figure_data(fig):
    """Same queryable view, straight from a live Figure (no file needed)."""
    return _scene.figure_data(fig)


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
        'timestamp': datetime.now().isoformat(timespec='seconds'),
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
    gridspec, twins, colors) — unlike replot_scene(), which rebuilds from
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
            f"the .mplscene via replot_scene().", stacklevel=2)

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
            saved.append(str(save_scene_file(fig, folder / stem)))
        except Exception as e:
            _loud(f"scene extraction failed: {e}")
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
    stamp = datetime.now().isoformat(timespec='seconds')
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
            saved.append(str(save_scene_file(fig, folder / stem)))
        except Exception as e:
            _loud(f"scene extraction failed: {e}")
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
        tk.Checkbutton(opt, text="Also save plot data (.mplscene)", variable=v_data).pack(anchor='w', padx=10, pady=4)
        tk.Checkbutton(opt, text="Also save editable figure (.mplpkl)", variable=v_pkl).pack(anchor='w', padx=10, pady=4)
        proceed = {'ok': False}
        tk.Button(opt, text="Choose location…",
                  command=lambda: (proceed.update(ok=True), opt.destroy())).pack(pady=6)
        opt.protocol("WM_DELETE_WINDOW", opt.destroy)
        root.wait_window(opt)
        if not proceed['ok']:
            root.destroy(); return None
        initial = (_widget_save_dir or Path.cwd())
        all_fmts = ['pdf', 'svg', 'png', 'jpg', 'jpeg', 'tif', 'tiff']
        fname = filedialog.asksaveasfilename(
            initialdir=str(initial),
            initialfile=(_figure_name_hint(fig) or 'figure') + f".{default_fmt}",
            defaultextension=f".{default_fmt}",
            filetypes=[(f"{default_fmt.upper()}", f"*.{default_fmt}")] + list((f"{a.upper()}", f"*.{a}") for a in list(set(sorted(all_fmts))-set([default_fmt]))) + [("All files", "*.*")])
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
        if p.suffix in ('.mplpkl', SCENE_EXT):
            return result
        if _prefs.get('save_pickle', True):
            savefig_bundle(self, p.with_suffix('.mplpkl'))
        if _prefs.get('save_data', True):
            try:
                save_scene_file(self, p.with_suffix(SCENE_EXT))
            except Exception as e:
                _loud(f"auto {SCENE_EXT} failed: {e}")
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
            cb_data = QtWidgets.QCheckBox("Also save plot data (.mplscene)")
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
        self.metadata: dict = {'created': datetime.now().isoformat(timespec='seconds'),
                               'notes': notes}

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
        self.metadata['saved'] = datetime.now().isoformat(timespec='seconds')
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
