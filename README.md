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

It behaves like a **dict of series**, so the idioms are the ones you already use:

```python
d = mfs.read_data("run.mplscene")

d.keys()                          # ['signal', 'band', 'pts']
'signal' in d                     # True
for label, s in d.items():
    print(label, len(s))

s = d["signal"]                   # one series
x, y = s.x, s.y                   # plain numpy arrays
x, y = s                          # or unpack, if you prefer
plt.plot(x, y)                    # ordinary matplotlib from here on

s["x"], s.keys()                  # dict-style too
d.to_dict()                       # {label: {'x': array, 'y': array}}
d.to_csv("run.csv")               # or d.to_dataframe() if pandas installed

[s for s in d if s.kind == "fill"]        # filter with a comprehension
band = d["band"]; band.lo, band.hi        # extras only where they exist
```

Nothing forces you to learn a new plotting vocabulary — pull out `x` and `y`
and use matplotlib exactly as you normally would.

## Settings

One validated object, in the spirit of `matplotlib.rcParams`:

```python
mfs.settings.save_pickle = False        # stop writing the .mplpkl
mfs.settings.save_data = True           # keep writing the .mplscene

with mfs.settings(save_data=False):     # or switch off for one block only
    fig.savefig("quick.png")

mfs.settings                            # print everything
mfs.settings.reset()
```

Typos fail loudly instead of doing nothing (`mfs.settings.save_picke = False`
raises and suggests `save_pickle`), and `"False"` is rejected rather than being
treated as true. `install(...)` and `configure(...)` accept the same options and
forward to this object.

## Resizing and font sizes

matplotlib keeps `figsize` (inches) and all sizes (points) independent. That is
right for publication work, where 8 pt text must stay 8 pt, so **nothing is
scaled automatically**. When you do want everything to scale together, it is
explicit:

```python
# reopen a saved figure at another size, keeping its proportions
fig = mfs.replot_scene("run.mplscene", figsize=(12, 8))            # scale="auto"
fig = mfs.replot_scene("run.mplscene", figsize=(12, 8), scale=False)  # size only

# or on any live figure
mfs.resize_figure(fig, (12, 8))     # set size and scale contents together
mfs.scale_figure(fig, 1.5)          # explicit factor
mfs.scale_figure(fig)               # fit sizes to the figure's current size
```

`"auto"` scales by `min(new_w/old_w, new_h/old_h)`, so a figure that only gets
wider does not overflow. Scaling covers fonts, line widths, marker sizes,
tick length/width/pad, spines, grid, legend handles and arrow heads.
`scale_figure(fig)` with no factor is relative to matplotlib's 6.4x4.8 design
size and is idempotent; the applied scale is stored in the `.mplscene`, so
reopening and calling it again does not scale twice.

Font sizes, weights and families of titles (all three positions), axis labels,
the legend and the suptitle are stored explicitly, so a figure reopens with the
same text sizes even if your current `rcParams` are different.

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
