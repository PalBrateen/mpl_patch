# mpl_figure_saver

MATLAB-`.fig`-style saving for matplotlib. Every time you save a figure — with
`fig.savefig()` or the toolbar **Save** button — it also drops two sidecar files
so you can reopen and *edit* the figure, or recover the underlying arrays, long
after the analysis has scrolled off your screen.

| file | what it is | reopen with | survives matplotlib upgrades? |
|------|------------|-------------|-------------------------------|
| `plot.png` / `.pdf` / … | the image you asked for | — | — |
| `plot.mplpkl` | pickle of the **exact** figure | `loadfig()` | ✗ (pickle) |
| `plot.mplscene` | JSON scene tree + **arrays** (no pickle) | `replot_scene()` / `read_data()` | ✓ |

The pickle gives you a fast, pixel-perfect reopen; the `.npz` is a version-proof
fallback (plain NumPy) that can rebuild the figure or just hand back the numbers.

## Install

Drop `mpl_figure_saver.py` **and** `mpl_scene.py` next to your code (no install step).
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

fig2, extra = mfs.loadfig("figs/sine.mplpkl")      # exact, editable figure
fig3 = mfs.replot_scene("figs/sine.mplscene")      # version-proof rebuild

d = mfs.read_data("figs/sine.mplscene")            # queryable data view
d.labels                                           # ['sin']
x, y = d["sin"]                                    # index by label, unpack
d.to_csv("sine.csv")                               # or d.to_dataframe()
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

v4 walks the artist tree **generically** instead of hand-coding each plot type,
so coverage is no longer a per-type allowlist. Verified pixel-identical on a
14-figure battery: lines, scatter, errorbars, bars, images, **`pcolormesh`**,
**`contourf`**, **`stackplot`**, `fill_between`, log scales, `twinx`/`twiny`,
2×2 grids, `axvline`/`axhline`/text, and minor-tick grids.

Known gaps: 3D axes, colorbars, polar projections, and custom `FuncFormatter`
lambdas (these degrade with a warning, never silently). Some artists come back
as an equivalent class (e.g. a filled contour as a `PathCollection`) — visually
identical, reported in the scene's `report` field. For a byte-exact copy on the
same matplotlib version, use `loadfig()` on the `.mplpkl`.

### Getting your data back

`read_data()` returns a queryable view — the part MATLAB's `.fig` has no API for:

```python
d = mfs.read_data("run.mplscene")
d.labels                      # ['signal', 'band', 'pts']
x, y = d["signal"]            # index by label, unpacks to arrays
d["band"].hi                  # kind-specific extras (lo/hi, xerr/yerr, sizes)
d.of_kind("fill")             # filter
d.to_csv("run.csv"); d.to_dataframe()
d["signal"].plot(ax)          # re-draw one series anywhere
```

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
unpickling runs arbitrary code.

`.mplscene` contains **no pickle**: a JSON tree plus arrays loaded with
`allow_pickle=False`. It is safe to open from any source, and readable from
any language. (v3's `.npz` did require `allow_pickle=True` — that is fixed.)

## License

MIT.
