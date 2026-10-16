# mpl_figure_saver

MATLAB-`.fig`-style saving for matplotlib. Every time you save a figure — with
`fig.savefig()` or the toolbar **Save** button — it also drops two sidecar files
so you can reopen and *edit* the figure, or recover the underlying arrays, long
after the analysis has scrolled off your screen.

| file | what it is | reopen with | survives matplotlib upgrades? |
|------|------------|-------------|-------------------------------|
| `plot.png` / `.pdf` / … | the image you asked for | — | — |
| `plot.mplpkl` | pickle of the **exact** figure | `loadfig()` | ✗ (pickle) |
| `plot_data.npz` | the **arrays** behind every artist | `replot_from_data()` / `load_arrays()` | ✓ |

The pickle gives you a fast, pixel-perfect reopen; the `.npz` is a version-proof
fallback (plain NumPy) that can rebuild the figure or just hand back the numbers.

## Install

Drop `mpl_figure_saver.py` next to your code (single file, no install step).
Requires `numpy` and `matplotlib`; `ipympl` only if you use `%matplotlib widget`.

```python
import mpl_figure_saver as mfs
mfs.install(save_dir="/path/to/figures", widget_mode="timestamp")
```

`install()` patches `Figure.savefig` and the Qt / ipympl toolbar **Save**
buttons. It's safe to re-run (works under `%autoreload`).

## Quick start

```python
import matplotlib.pyplot as plt, numpy as np, mpl_figure_saver as mfs
mfs.install(save_dir="figs", widget_mode="timestamp")

x = np.linspace(0, 10, 200)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), label="sin")
ax.legend()
fig.savefig("figs/sine.png")          # also writes sine.mplpkl + sine_data.npz

fig2, extra = mfs.loadfig("figs/sine.mplpkl")   # exact, editable figure
fig3 = mfs.replot_from_data("figs/sine_data.npz")  # version-proof rebuild
x, y = mfs.load_arrays("figs/sine_data.npz")[0]["lines"][0].values()  # raw arrays
```

See **`demo_mpl_figure_saver.ipynb`** (or run `python demo_mpl_figure_saver.py`)
for a full tour, including the algorithm notes.

## Backends & where saves go

| context | **Save** button behaviour |
|---------|---------------------------|
| Qt (`%matplotlib qt`) | dialog with two checkboxes (`.npz`, `.mplpkl`); writes a `<name>/` folder. Untick both → plain image. |
| ipympl, `widget_mode="dialog"` | local Python save dialog; **nothing written if you cancel**. Local / remote-desktop only. |
| ipympl, `widget_mode="timestamp"` | writes `plot_<timestamp>[_suptitle]/` into `save_dir`; browser download still happens. The right choice over **SSH / headless**. |
| `fig.savefig(path)` | image + `.npz` + `.mplpkl` next to `path`. |

Per-machine defaults can come from environment variables `MPL_FIG_SAVER_DIR`
and `MPL_FIG_SAVER_MODE`, so one notebook runs unchanged on laptop, remote
desktop and SSH.

**VS Code tip:** call `mfs.fix_vscode_widget_style()` once so `%matplotlib
widget` figures follow your theme instead of rendering on a white box.

## What round-trips through the `.npz`

Reconstructed by `replot_from_data()`: lines (labeled *or* not), scatter,
errorbars, bars, images, `axvline`/`axhline`, text, scales, limits, legend,
label/tick colors, figure size, **`fill_between`/`fill_betweenx`** (incl.
`where=` gaps), **`twinx`/`twiny`**, and regular subplot grids (e.g. 2×2).

Not reconstructed from arrays (use `loadfig()` on the `.mplpkl` for these):
`pcolormesh`/contour/stackplot and irregular `gridspec` layouts.

## `ExperimentResult`

Bundle raw data, parameters and figures into one `.expr` file for archiving:

```python
r = mfs.ExperimentResult("ESR #42", params={"B_mT": 12.5})
r.store("freq", freq); r.store("signal", sig)
r.attach_figure(fig)
r.save("esr_042.expr")
r = mfs.ExperimentResult.load("esr_042.expr")
```

## Security

`.mplpkl` and `.expr` use `pickle` — **only load files you trust**, since
unpickling runs arbitrary code. The `.npz` is plain arrays and always safe.

## License

MIT.
