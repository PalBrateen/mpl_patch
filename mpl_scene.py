"""P0 spike: generic matplotlib scene-graph dump / replay (no pickle)."""
from __future__ import annotations
import json, zipfile, io, warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.artist import ArtistInspector
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from matplotlib.text import Text, Annotation
from matplotlib.patches import Patch, Rectangle, Polygon, PathPatch, FancyArrowPatch
from matplotlib.collections import (Collection, PolyCollection, LineCollection,
                                    PathCollection, QuadMesh)
from matplotlib.image import AxesImage
from matplotlib.spines import Spine
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.dates as mdates

SCHEMA = 1

# ---------------------------------------------------------------- array store
class _Arrays:
    def __init__(self): self.d = {}; self.n = 0
    def put(self, a):
        a = np.asarray(a)
        if a.dtype == object:            # object arrays can't be pickle-free
            return None
        k = f"a{self.n}"; self.n += 1; self.d[k] = a
        return {"__arr__": k}
    def get(self, ref): return self.d[ref["__arr__"]]

# ------------------------------------------------------------ value encoding
_SKIP_PROPS = {
    'figure', 'axes', 'children', 'transform', 'clip_box', 'clip_path',
    'tightbbox', 'window_extent', 'picker', 'contains', 'offset_transform',
    'xdata', 'ydata',                      # covered by 'data'
    'data_transform', 'patch_transform', 'verts', 'verts_and_codes',
    'subplotspec', 'gridspec', 'shared_x_axes', 'shared_y_axes', 'position',
    'cursor_data', 'xydata', 'xy',         # annotation xy handled explicitly
    'datalim', 'default_bbox_extra_artists', 'images', 'lines',
    'legend', 'legend_handles_labels', 'xaxis', 'yaxis', 'xaxis_transform',
    'yaxis_transform', 'xticklines', 'yticklines', 'xgridlines', 'ygridlines',
    'xticklabels', 'yticklabels', 'xmajorticklabels', 'xminorticklabels',
    'ymajorticklabels', 'yminorticklabels', 'xticks', 'yticks', 'xbound',
    'ybound', 'navigate_mode', 'axisbelow', 'anchor', 'frame_on',
    'xlim', 'ylim', 'xscale', 'yscale', 'title', 'xlabel', 'ylabel',
    'aspect', 'adjustable', 'box_aspect', 'facecolor', 'fc',
    'prop_cycle', 'axis_on', 'axis_off', 'canvas', 'dpi', 'size_inches',
    'tight_layout', 'constrained_layout', 'constrained_layout_pads',
    'layout_engine', 'figwidth', 'figheight', 'frameon', 'edgecolor',
    'linewidth', 'clip_on',                 # re-added per-class below if needed
    'array', 'paths', 'offsets', 'coordinates', 'segments',
    'path', 'path_effects', 'snap', 'sketch_params', 'in_layout',
    'rasterized', 'agg_filter', 'url', 'gid', 'label',  # label handled explicitly
    'mouseover', 'animated', 'zorder', 'visible', 'alpha',  # re-added explicitly
}
# Properties we DO want on leaf artists even though they're in the broad skip set
_KEEP = {'alpha', 'zorder', 'visible', 'label', 'linewidth', 'clip_on',
         'edgecolor', 'facecolor',
         # patch geometry (Rectangle/Ellipse/Wedge/Polygon reconstruct from these)
         'xy', 'x', 'y', 'width', 'height', 'angle', 'center', 'radius',
         'theta1', 'theta2', 'bounds'}

def _hex(c):
    """Canonical colour string, so dump(replay(dump(f))) is a true fixed point."""
    try: return mcolors.to_hex(c, keep_alpha=True)
    except Exception: return c


def _text_style(t):
    """Font-related style of a Text artist (size/weight/style/family/color...).

    Stored explicitly so a replay does not silently fall back to whatever
    rcParams happen to be active when the figure is reopened.
    """
    try:
        fam = t.get_fontfamily()
        return {"size": float(t.get_fontsize()),
                "weight": t.get_fontweight(),
                "style": t.get_fontstyle(),
                "family": list(fam) if isinstance(fam, (list, tuple)) else fam,
                "color": mcolors.to_hex(t.get_color(), keep_alpha=True),
                "rotation": float(t.get_rotation())}
    except Exception:
        return {}


def _apply_text_style(t, st):
    if not st: return
    for k, meth in (("size", "set_fontsize"), ("weight", "set_fontweight"),
                    ("style", "set_fontstyle"), ("family", "set_fontfamily"),
                    ("color", "set_color"), ("rotation", "set_rotation")):
        if st.get(k) is not None:
            try: getattr(t, meth)(st[k])
            except Exception: pass


def _tf_tag(tr, ax, fig):
    if ax is not None:
        if tr is ax.transData: return "data"
        if tr is ax.transAxes: return "axes"
        if tr is ax.get_xaxis_transform(): return "xaxis"
        if tr is ax.get_yaxis_transform(): return "yaxis"
        # axhspan/axvspan wrap a blended transform in a composite, so identity
        # fails. Compare what the transforms actually DO on probe points.
        try:
            probe = np.array([[0.0, 0.0], [1.0, 1.0], [0.3, 0.7]])
            got = tr.transform(probe)
            for tag, cand in (("xaxis", ax.get_xaxis_transform()),
                              ("yaxis", ax.get_yaxis_transform()),
                              ("axes", ax.transAxes), ("data", ax.transData)):
                if np.allclose(got, cand.transform(probe), rtol=1e-9, atol=1e-9):
                    return tag
        except Exception:
            pass
    if fig is not None and tr is fig.transFigure: return "figure"
    return None

def _enc(v, arrays, report, path=""):
    if v is None or isinstance(v, (bool, int, float, str)): return v
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            report.append(f"object-array @{path}"); return None
        return arrays.put(v)
    if isinstance(v, MPath):
        return {"__path__": {"v": arrays.put(v.vertices),
                             "c": (arrays.put(v.codes) if v.codes is not None else None)}}
    if isinstance(v, mcolors.Colormap): return {"__cmap__": v.name}
    if isinstance(v, mcolors.Normalize):
        return {"__norm__": type(v).__name__,
                "vmin": None if v.vmin is None else float(v.vmin),
                "vmax": None if v.vmax is None else float(v.vmax)}
    if isinstance(v, (list, tuple)):
        out = [_enc(x, arrays, report, path) for x in v]
        if any(x is None and y is not None for x, y in zip(out, v)): 
            return None
        return out
    if isinstance(v, dict):
        return {str(k): _enc(x, arrays, report, f"{path}.{k}") for k, x in v.items()}
    report.append(f"{type(v).__name__} @{path}")
    return None

def _dec(v, arrays):
    if isinstance(v, dict):
        if "__arr__" in v: return arrays.get(v)
        if "__path__" in v:
            p = v["__path__"]
            return MPath(arrays.get(p["v"]),
                         arrays.get(p["c"]) if p["c"] else None)
        if "__cmap__" in v: return matplotlib.colormaps[v["__cmap__"]]
        if "__norm__" in v:
            cls = getattr(mcolors, v["__norm__"], mcolors.Normalize)
            return cls(vmin=v["vmin"], vmax=v["vmax"])
        return {k: _dec(x, arrays) for k, x in v.items()}
    if isinstance(v, list): return [_dec(x, arrays) for x in v]
    return v

# ------------------------------------------------------- locator / formatter
def _enc_locator(loc):
    t = type(loc).__name__
    if isinstance(loc, mticker.FixedLocator):
        return {"cls": t, "locs": np.asarray(loc.locs).tolist()}
    if isinstance(loc, mticker.MultipleLocator):
        try: return {"cls": t, "base": float(loc._edge.step)}
        except Exception: return {"cls": "AutoLocator"}
    if isinstance(loc, (mticker.MaxNLocator,)):
        return {"cls": t, "nbins": getattr(loc, "_nbins", "auto")}
    if isinstance(loc, mticker.LogLocator):
        return {"cls": t, "base": float(getattr(loc, "_base", 10))}
    if isinstance(loc, mdates.AutoDateLocator): return {"cls": "AutoDateLocator"}
    if isinstance(loc, mdates.DayLocator): return {"cls": "DayLocator"}
    if isinstance(loc, mdates.MonthLocator): return {"cls": "MonthLocator"}
    if isinstance(loc, mdates.YearLocator): return {"cls": "YearLocator"}
    if isinstance(loc, (mticker.AutoLocator, mticker.AutoMinorLocator,
                        mticker.NullLocator, mticker.IndexLocator)):
        return {"cls": t}
    return {"cls": "AutoLocator"}

def _dec_locator(spec):
    cls = spec.get("cls", "AutoLocator")
    try:
        if cls == "FixedLocator": return mticker.FixedLocator(spec["locs"])
        if cls == "MultipleLocator": return mticker.MultipleLocator(spec["base"])
        if cls == "MaxNLocator": return mticker.MaxNLocator(nbins=spec.get("nbins", "auto"))
        if cls == "LogLocator": return mticker.LogLocator(base=spec.get("base", 10))
        if cls == "NullLocator": return mticker.NullLocator()
        if cls == "AutoMinorLocator": return mticker.AutoMinorLocator()
        if cls == "AutoDateLocator": return mdates.AutoDateLocator()
        if cls == "DayLocator": return mdates.DayLocator()
        if cls == "MonthLocator": return mdates.MonthLocator()
        if cls == "YearLocator": return mdates.YearLocator()
    except Exception: pass
    return mticker.AutoLocator()

def _enc_formatter(fm):
    t = type(fm).__name__
    if isinstance(fm, mdates.ConciseDateFormatter): return {"cls": "ConciseDateFormatter"}
    if isinstance(fm, mdates.AutoDateFormatter): return {"cls": "AutoDateFormatter"}
    if isinstance(fm, mdates.DateFormatter): return {"cls": "DateFormatter", "fmt": fm.fmt}
    if isinstance(fm, mticker.FixedFormatter):
        return {"cls": t, "seq": list(fm.seq)}
    if isinstance(fm, mticker.StrMethodFormatter):
        return {"cls": t, "fmt": fm.fmt}
    if isinstance(fm, mticker.FormatStrFormatter):
        return {"cls": t, "fmt": fm.fmt}
    if isinstance(fm, (mticker.ScalarFormatter, mticker.LogFormatterSciNotation,
                       mticker.LogFormatter, mticker.NullFormatter,
                       mticker.PercentFormatter, mticker.EngFormatter)):
        return {"cls": t}
    return {"cls": "ScalarFormatter", "lossy": t}   # e.g. FuncFormatter(lambda)

def _dec_formatter(spec):
    cls = spec.get("cls", "ScalarFormatter")
    try:
        if cls == "FixedFormatter": return mticker.FixedFormatter(spec["seq"])
        if cls == "StrMethodFormatter": return mticker.StrMethodFormatter(spec["fmt"])
        if cls == "FormatStrFormatter": return mticker.FormatStrFormatter(spec["fmt"])
        if cls == "NullFormatter": return mticker.NullFormatter()
        if cls == "DateFormatter": return mdates.DateFormatter(spec["fmt"])
        if cls in ("AutoDateFormatter", "ConciseDateFormatter"):
            loc = mdates.AutoDateLocator()
            return (mdates.ConciseDateFormatter(loc) if cls.startswith("Concise")
                    else mdates.AutoDateFormatter(loc))
        f = getattr(mticker, cls, None)
        if f is not None: return f()
    except Exception: pass
    return mticker.ScalarFormatter()

# --------------------------------------------------------------- leaf artists
def _props_of(art, arrays, report):
    insp = ArtistInspector(art)
    setters = set(insp.get_setters())
    out = {}
    try: props = art.properties()
    except Exception: props = {}
    for k, v in props.items():
        if k not in setters: continue
        if k in _SKIP_PROPS and k not in _KEEP: continue
        e = _enc(v, arrays, report, f"{type(art).__name__}.{k}")
        if e is None and v is not None: continue
        out[k] = e
    return out

def _base_of(art):
    if isinstance(art, Line2D): return "line"
    if isinstance(art, AxesImage): return "image"
    if isinstance(art, QuadMesh): return "quadmesh"
    if isinstance(art, Collection): return "collection"
    if isinstance(art, Patch): return "patch"
    if isinstance(art, Text): return "text"
    return "artist"


def _numeric(vals):
    """Coerce plot data to float, converting datetimes via date2num."""
    a = np.asarray(vals)
    if a.dtype.kind in "Mm" or (a.dtype == object and a.size
                                and hasattr(a.flat[0], "toordinal")):
        return np.asarray(mdates.date2num(a), dtype=np.float64)
    return np.asarray(a, dtype=np.float64)


def _errs(cont, arrays):
    """Per-axis error magnitudes from an ErrorbarContainer -> {'x':..,'y':..}."""
    out = {}
    barlines = cont[2] if len(cont) > 2 else ()
    dl = cont[0]
    for axis, want in (("x", cont.has_xerr), ("y", cont.has_yerr)):
        if not want or not barlines:
            out[axis] = None; continue
        try:
            if cont.has_xerr and cont.has_yerr:
                bc = barlines[0] if axis == "x" else barlines[1]
            else:
                bc = barlines[0]
            segs = bc.get_segments()
            comp = 0 if axis == "x" else 1
            base = np.asarray(dl.get_xdata() if axis == "x" else dl.get_ydata(), float)
            lo = np.array([sg[0][comp] for sg in segs], float)
            hi = np.array([sg[1][comp] for sg in segs], float)
            out[axis] = arrays.put(np.column_stack([base - lo, hi - base]))
        except Exception:
            out[axis] = None
    return out


def _dump_artist(art, ax, fig, arrays, report):
    node = {"cls": type(art).__name__, "base": _base_of(art),
            "props": _props_of(art, arrays, report),
            "label": art.get_label()}
    _tr = (art.get_data_transform() if isinstance(art, Patch)
           else art.get_transform())
    tr = _tf_tag(_tr, ax, fig)
    if tr: node["tf"] = tr
    # class-specific payload the generic props skip
    if isinstance(art, Line2D):
        node["data"] = [arrays.put(_numeric(art.get_xdata())),
                        arrays.put(_numeric(art.get_ydata()))]
    elif isinstance(art, QuadMesh):
        try:
            node["coords"] = arrays.put(np.asarray(art.get_coordinates(), float))
            A = art.get_array()
            node["array"] = arrays.put(np.asarray(A, float))
            node["shading"] = "flat"
            cm = art.get_cmap()
            node["cmap"] = cm.name if cm else None
            node["clim"] = [float(c) for c in art.get_clim()]
        except Exception as e: report.append(f"QuadMesh payload: {e}")
    elif isinstance(art, AxesImage):
        node["array"] = arrays.put(np.asarray(art.get_array(), float))
        node["extent"] = list(art.get_extent())
        cm = art.get_cmap(); node["cmap"] = cm.name if cm else None
        node["clim"] = [float(c) for c in art.get_clim()]
    elif isinstance(art, Collection):
        node["paths"] = [_enc(p, arrays, report) for p in art.get_paths()]
        try:                                  # no fill at all => stroke-only
            fc_ = np.asarray(art.get_facecolor())
            node["line_like"] = bool(fc_.size == 0 or np.all(fc_[:, -1] == 0))
        except Exception:
            node["line_like"] = False
        off = art.get_offsets()
        if off is not None and len(np.asarray(off)): 
            node["offsets"] = arrays.put(np.asarray(off, float))
        for key, fn in (("fc", art.get_facecolor), ("ec", art.get_edgecolor),
                        ("lw", art.get_linewidth)):
            try: node[key] = _enc(np.asarray(fn()), arrays, report)
            except Exception: pass
        try:
            A = art.get_array()
            if A is not None: node["array"] = arrays.put(np.asarray(A, float))
        except Exception: pass
        try:
            cm = art.get_cmap()
            if cm is not None: node["cmap"] = cm.name
            if art.get_array() is not None: node["clim"] = [float(c) for c in art.get_clim()]
        except Exception: pass
        try:
            s_ = art.get_sizes()
            if s_ is not None and len(np.asarray(s_)): node["sizes"] = arrays.put(np.asarray(s_, float))
        except Exception: pass
    elif isinstance(art, Patch):
        node["path"] = _enc(art.get_path().transformed(art.get_patch_transform()),
                            arrays, report)
    elif isinstance(art, Text):
        node["text"] = art.get_text()
        node["pos"] = [float(c) for c in art.get_position()]
        if isinstance(art, Annotation):
            try:
                node["anno_xy"] = [float(c) for c in art.xy]
                ap = art.arrowprops
                if ap: node["arrowprops"] = {k: v for k, v in ap.items()
                                             if isinstance(v, (str, int, float, bool))}
            except Exception: pass
    return node

_FALLBACK = {}

def _make_artist(node, ax, arrays, report):
    cls = node["cls"]; base = node.get("base", ""); props = node.get("props", {})
    try:
        if base == "line":
            x = arrays.get(node["data"][0]); y = arrays.get(node["data"][1])
            return Line2D(x, y)
        if base == "text":
            if node.get("anno_xy") is not None:
                an = ax.annotate(node.get("text", ""), xy=node["anno_xy"],
                                 xytext=node["pos"],
                                 arrowprops=node.get("arrowprops") or None)
                return ("attached", an)
            return Text(node["pos"][0], node["pos"][1], node.get("text", ""))
        if base == "quadmesh" and "coords" in node:
            c = arrays.get(node["coords"]); A = arrays.get(node["array"])
            qm = ax.pcolormesh(c[..., 0], c[..., 1], A.reshape(c.shape[0]-1, c.shape[1]-1),
                               cmap=node.get("cmap"))
            if node.get("clim"): qm.set_clim(node["clim"])
            return ("attached", qm)
        if base == "image":
            im = ax.imshow(arrays.get(node["array"]), extent=node.get("extent"),
                           cmap=node.get("cmap"))
            if node.get("clim"): im.set_clim(node["clim"])
            return ("attached", im)
        if base == "collection" and "offsets" in node and "sizes" in node:
            # scatter-like: offsets in data coords, marker path scaled by sizes.
            off = arrays.get(node["offsets"]); sz = arrays.get(node["sizes"])
            fc = _dec(node.get("fc"), arrays)
            sc = ax.scatter(off[:, 0], off[:, 1], s=sz,
                            c=(fc if fc is not None and len(np.asarray(fc)) else None))
            return ("attached", sc)
        if base == "collection" and node.get("line_like"):
            # Stroke-only (contour lines, LineCollection): keep the Path objects
            # so multi-part geometry is exact, and draw with no fill.
            paths = [_dec(p, arrays) for p in node.get("paths", [])]
            lc = PathCollection(list(paths))
            lc.set_facecolor("none")
            ec = _dec(node.get("ec"), arrays)
            if ec is not None and len(np.asarray(ec)): lc.set_edgecolor(ec)
            lw = _dec(node.get("lw"), arrays)
            if lw is not None and len(np.atleast_1d(lw)): lc.set_linewidth(lw)
            lc._mfs_done = True
            return lc
        if base == "collection":                     # filled areas, contours, ...
            paths = [_dec(p, arrays) for p in node.get("paths", [])]
            # PathCollection keeps Path codes, so holes in filled contours and
            # multi-part fill_between(where=) regions survive the round trip.
            c = PathCollection(list(paths))
            if "offsets" in node:
                try: c.set_offsets(arrays.get(node["offsets"]))
                except Exception: pass
            if cls not in ("PolyCollection", "FillBetweenPolyCollection", "QuadContourSet"):
                _FALLBACK[cls] = _FALLBACK.get(cls, 0) + 1
                report.append(f"visual-fallback PolyCollection for {cls}")
            return c
        if base == "patch":
            import matplotlib.patches as mpatches
            ctor = {"Rectangle": lambda: mpatches.Rectangle((0, 0), 1, 1),
                    "Circle": lambda: mpatches.Circle((0, 0), 1),
                    "Ellipse": lambda: mpatches.Ellipse((0, 0), 1, 1),
                    "Wedge": lambda: mpatches.Wedge((0, 0), 1, 0, 90),
                    "Polygon": lambda: mpatches.Polygon([[0, 0], [1, 1]]),
                    "FancyBboxPatch": lambda: mpatches.FancyBboxPatch((0, 0), 1, 1),
                    }.get(cls)
            if ctor is not None:
                try: return ctor()               # props below restore geometry
                except Exception: pass
            p = _dec(node["path"], arrays)
            if cls != "PathPatch":
                _FALLBACK[cls] = _FALLBACK.get(cls, 0) + 1
                report.append(f"visual-fallback PathPatch for {cls}")
            return PathPatch(p)
    except Exception as e:
        report.append(f"make {cls} failed: {type(e).__name__}: {e}")
        return None
    report.append(f"unknown class {cls}")
    return None

def _apply_payload(art, node, arrays, report):
    """Apply the class-specific payload that generic props deliberately skip."""
    if getattr(art, "_mfs_done", False):
        return                      # stroke-only collection already styled
    rep = getattr(art, "_mfs_rep", None)
    for key, meth in (("fc", "set_facecolor"), ("ec", "set_edgecolor"),
                      ("lw", "set_linewidth"), ("sizes", "set_sizes")):
        if key in node and node[key] is not None:
            val = _dec(node[key], arrays)
            if rep is not None and key in ("fc", "ec", "lw"):
                a = np.asarray(val)
                if a.ndim >= 1 and len(a) == len(rep) and sum(rep) != len(rep):
                    val = np.repeat(a, rep, axis=0)   # one path -> many polygons
            try: getattr(art, meth)(val)
            except Exception as e: report.append(f"{meth}: {type(e).__name__}")
    if node.get("array") is not None and not isinstance(art, (AxesImage,)):
        try:
            art.set_array(_dec(node["array"], arrays))
            if node.get("cmap"): art.set_cmap(node["cmap"])
            if node.get("clim"): art.set_clim(node["clim"])
        except Exception as e: report.append(f"set_array: {type(e).__name__}")


def _apply_props(art, props, arrays, report):
    for k in sorted(props):
        v = _dec(props[k], arrays)
        try: getattr(art, "set_" + k)(v)
        except Exception as e:
            report.append(f"set_{k} on {type(art).__name__}: {type(e).__name__}")

# ------------------------------------------------------------- axis container
_AXIS_KEYS = ("scale", "lim", "label", "label_color", "autoscale",
              "major_loc", "major_fmt", "minor_loc", "minor_fmt",
              "tick_params", "grid")

def _dump_axis(ax, which):
    axis = ax.xaxis if which == "x" else ax.yaxis
    get = (ax.get_xscale, ax.get_xlim) if which == "x" else (ax.get_yscale, ax.get_ylim)
    d = {
        "scale": get[0](), "lim": [float(v) for v in get[1]()],
        "label": axis.get_label().get_text(),
        "label_color": _hex(axis.label.get_color()),
        "label_style": _text_style(axis.label),
        "autoscale": bool(ax.get_autoscalex_on() if which == "x" else ax.get_autoscaley_on()),
        "major_loc": _enc_locator(axis.get_major_locator()),
        "major_fmt": _enc_formatter(axis.get_major_formatter()),
        "minor_loc": _enc_locator(axis.get_minor_locator()),
        "minor_fmt": _enc_formatter(axis.get_minor_formatter()),
    }
    try:
        conv = axis.get_converter() if hasattr(axis, "get_converter") else None
        d["is_date"] = bool(conv is not None and isinstance(
            conv, (mdates.DateConverter, getattr(mdates, "_SwitchableDateConverter", ()))))
    except Exception:
        d["is_date"] = False
    try: d["tick_params"] = {k: v for k, v in axis.get_tick_params().items()
                             if isinstance(v, (bool, int, float, str))
                             and k not in ("labelcolor", "color")}
    except Exception: d["tick_params"] = {}
    try:
        tl = axis.get_majorticklabels()
        d["tick_color"] = _hex(tl[0].get_color()) if tl else None
    except Exception: d["tick_color"] = None
    return d

def _apply_axis(ax, which, d, report):
    axis = ax.xaxis if which == "x" else ax.yaxis
    try:
        (ax.set_xscale if which == "x" else ax.set_yscale)(d["scale"])
    except Exception as e: report.append(f"{which}scale: {e}")
    if d.get("is_date"):
        try: axis.axis_date()
        except Exception: pass
    axis.set_major_locator(_dec_locator(d["major_loc"]))
    axis.set_major_formatter(_dec_formatter(d["major_fmt"]))
    axis.set_minor_locator(_dec_locator(d["minor_loc"]))
    axis.set_minor_formatter(_dec_formatter(d["minor_fmt"]))
    (ax.set_xlabel if which == "x" else ax.set_ylabel)(
        d.get("label", ""), color=d.get("label_color") or "black")
    _apply_text_style(axis.label, d.get("label_style"))
    if d.get("tick_params"):
        try: ax.tick_params(axis=which, **d["tick_params"])
        except Exception: pass
    if d.get("tick_color"):
        try: ax.tick_params(axis=which, labelcolor=d["tick_color"])
        except Exception: pass
    # limits last, and restore the auto/explicit MODE (MATLAB's XLimMode).
    # NOTE: the autoscale flag itself is restored by the caller in a final pass,
    # because a later set_xlim on a SHARED/twinned axes clears it again.
    (ax.set_xlim if which == "x" else ax.set_ylim)(d["lim"])

# --------------------------------------------------------------------- figure
def dump_figure(fig):
    arrays = _Arrays(); report = []
    axes = list(fig.axes)
    idx = {id(a): i for i, a in enumerate(axes)}
    # twin / share groups
    def grp(a, which):
        g = a.get_shared_x_axes() if which == "x" else a.get_shared_y_axes()
        return sorted(idx[id(s)] for s in g.get_siblings(a) if id(s) in idx)
    tree = {"schema": SCHEMA, "mpl": matplotlib.__version__,
            "np": np.__version__,
            "figure": {"size": [float(v) for v in fig.get_size_inches()],
                       "dpi": float(fig.dpi),
                       "facecolor": mcolors.to_hex(fig.get_facecolor(), keep_alpha=True),
                       "suptitle": fig._suptitle.get_text() if fig._suptitle else "",
                       "suptitle_style": (_text_style(fig._suptitle)
                                          if fig._suptitle else {}),
                       "mfs_scale": float(getattr(fig, "_mfs_scale", 1.0))},
            "axes": []}
    cbar_of = {}
    for i, ax in enumerate(axes):
        info = getattr(ax, "_colorbar_info", None)
        if info is not None or hasattr(ax, "_colorbar"):
            cb = getattr(ax, "_colorbar", None)
            try:
                mp_ax = cb.mappable.axes
                cbar_of[i] = {"parent": idx.get(id(mp_ax)),
                              "label": cb.ax.get_ylabel() or cb.ax.get_xlabel(),
                              "orientation": getattr(cb, "orientation", "vertical")}
            except Exception:
                cbar_of[i] = {"parent": None, "label": "", "orientation": "vertical"}
    tree["colorbars"] = cbar_of
    for i, ax in enumerate(axes):
        node = {"i": i,
                "pos": [float(v) for v in ax.get_position().bounds],
                "geom": None, "projection": ax.name,
                "title": ax.get_title(),
                "titles": {loc: {"text": ax.get_title(loc=loc),
                                 "style": _text_style({"center": ax.title,
                                                       "left": ax._left_title,
                                                       "right": ax._right_title}[loc])}
                           for loc in ("center", "left", "right")
                           if ax.get_title(loc=loc)},
                "facecolor": mcolors.to_hex(ax.get_facecolor(), keep_alpha=True),
                "aspect": (ax.get_aspect() if isinstance(ax.get_aspect(), str)
                           else float(ax.get_aspect())),
                "x": _dump_axis(ax, "x"), "y": _dump_axis(ax, "y"),
                "sharex": grp(ax, "x"), "sharey": grp(ax, "y"),
                "twin_group": sorted(idx[id(s)] for s in
                                     getattr(ax, "_twinned_axes", _Dummy()).get_siblings(ax)
                                     if id(s) in idx) if hasattr(ax, "_twinned_axes") else [i],
                "legend": None, "artists": []}
        try:
            ss = ax.get_subplotspec()
            if ss is not None: node["geom"] = list(ss.get_geometry()[:4])
        except Exception: pass
        node["containers"] = []
        for cont in getattr(ax, "containers", []):
            cd = {"cls": type(cont).__name__, "label": cont.get_label()}
            try:
                from matplotlib.container import ErrorbarContainer, BarContainer
                if isinstance(cont, ErrorbarContainer):
                    dl = cont[0]
                    cd["x"] = arrays.put(np.asarray(dl.get_xdata(), float))
                    cd["y"] = arrays.put(np.asarray(dl.get_ydata(), float))
                    cd["has_xerr"] = bool(cont.has_xerr)
                    cd["has_yerr"] = bool(cont.has_yerr)
                    cd["err"] = _errs(cont, arrays)
                    cd["fmt_line"] = id(dl)
                    dl0 = cont[0]
                    cd["style"] = {"color": dl0.get_color(),
                                   "marker": dl0.get_marker(),
                                   "linestyle": dl0.get_linestyle(),
                                   "markersize": float(dl0.get_markersize())}
                    try:
                        bl = cont[2][0]
                        cd["style"]["ecolor"] = _enc(np.asarray(bl.get_color())[0],
                                                     arrays, report)
                        cd["style"]["elinewidth"] = float(np.asarray(bl.get_linewidth())[0])
                    except Exception: pass
                    cd["capsize"] = (float(cont[1][0].get_markersize()) / 2
                                     if len(cont) > 1 and cont[1] else 0.0)
                elif isinstance(cont, BarContainer):
                    cd["values"] = arrays.put(np.asarray(cont.datavalues, float))
                    cd["orientation"] = cont.orientation
                    cd["x"] = arrays.put(np.asarray(
                        [p.get_x() for p in cont.patches], float))
                    cd["widths"] = arrays.put(np.asarray(
                        [p.get_width() for p in cont.patches], float))
                    cd["bottoms"] = arrays.put(np.asarray(
                        [p.get_y() for p in cont.patches], float))
                    if cont.patches:
                        cd["color"] = _enc(np.asarray(
                            mcolors.to_rgba(cont.patches[0].get_facecolor())),
                            arrays, report)
            except Exception as e:
                report.append(f"container {cd['cls']}: {e}")
            node["containers"].append(cd)
        leg = ax.get_legend()
        if leg is not None:
            node["legend"] = {"labels": [t.get_text() for t in leg.get_texts()],
                              "loc": (leg._loc if isinstance(leg._loc, (int, str))
                                      else "best"),
                              "ncols": int(getattr(leg, "_ncols", 1) or 1),
                              "frameon": bool(leg.get_frame_on()),
                              "title": leg.get_title().get_text(),
                              "text_style": (_text_style(leg.get_texts()[0])
                                             if leg.get_texts() else {}),
                              "title_style": _text_style(leg.get_title())}
        owned = set()
        for cont in getattr(ax, "containers", []):
            for ch in cont.get_children():
                owned.add(id(ch))
                for sub in getattr(ch, "get_children", lambda: [])():
                    owned.add(id(sub))
        for art in (list(ax.lines) + list(ax.collections) + list(ax.patches)
                    + list(ax.images) + list(ax.texts)):
            if id(art) in owned:       # drawn by its container on replay
                continue
            n = _dump_artist(art, ax, fig, arrays, report)
            if n: node["artists"].append(n)
        node["spines"] = {k: {"visible": bool(s.get_visible()),
                              "color": mcolors.to_hex(s.get_edgecolor(), keep_alpha=True),
                              "lw": float(s.get_linewidth())}
                          for k, s in ax.spines.items()}
        tree["axes"].append(node)
    tree["report"] = report
    return _scrub(tree), arrays


def _scrub(o):
    """Make the tree strictly JSON-safe (numpy scalars -> python)."""
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, np.ndarray): return [_scrub(x) for x in o.tolist()]
    if isinstance(o, dict): return {str(k): _scrub(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_scrub(v) for v in o]
    return o

class _Dummy:
    def get_siblings(self, a): return [a]

def _proj(node):
    p = node.get("projection")
    return None if p in (None, "rectilinear") else p


def _replay_container(ax, cd, arrays, report):
    """Rebuild an ErrorbarContainer / BarContainer with its native call."""
    cls = cd.get("cls")
    if cls == "ErrorbarContainer":
        x = arrays.get(cd["x"]); y = arrays.get(cd["y"])
        err = cd.get("err") or {}
        xe = arrays.get(err["x"]).T if err.get("x") is not None else None
        ye = arrays.get(err["y"]).T if err.get("y") is not None else None
        st = cd.get("style", {})
        fmt = f"{st.get('marker') or ''}{'' if st.get('linestyle') in ('None','') else ''}"
        kw = dict(xerr=xe, yerr=ye, label=cd.get("label", ""),
                  color=st.get("color"), marker=st.get("marker") or None,
                  linestyle=st.get("linestyle", "None"),
                  markersize=st.get("markersize"),
                  capsize=cd.get("capsize", 0.0))
        if st.get("ecolor") is not None: kw["ecolor"] = _dec(st["ecolor"], arrays)
        if st.get("elinewidth") is not None: kw["elinewidth"] = st["elinewidth"]
        ax.errorbar(x, y, **{k: v for k, v in kw.items() if v is not None})
    elif cls == "BarContainer":
        vals = arrays.get(cd["values"]); x = arrays.get(cd["x"])
        w = arrays.get(cd["widths"]); b = arrays.get(cd.get("bottoms")) if cd.get("bottoms") else None
        color = _dec(cd.get("color"), arrays)
        if cd.get("orientation") == "horizontal":
            ax.barh(x, vals, height=w, left=b, label=cd.get("label", ""),
                    color=color, align="edge")
        else:
            ax.bar(x, vals, width=w, bottom=b, label=cd.get("label", ""),
                   color=color, align="edge")
    else:
        report.append(f"unknown container {cls}")


def replay_figure(tree, arrays, figsize=None, scale="auto"):
    report = []
    f = tree["figure"]
    fig = plt.figure(figsize=f["size"], dpi=f["dpi"], facecolor=f["facecolor"])
    nodes = tree["axes"]
    made = {}
    # primaries first (an axes is a twin if it isn't the lowest index of its group)
    def is_twin(n): return n["twin_group"] and n["i"] != min(n["twin_group"])
    import matplotlib.gridspec as mgs
    gs_cache = {}
    for n in nodes:
        if is_twin(n): continue
        g = n.get("geom")
        if g:                                   # rebuild the real subplotspec
            key = (g[0], g[1])
            gs = gs_cache.get(key) or mgs.GridSpec(g[0], g[1], figure=fig)
            gs_cache[key] = gs
            ax = fig.add_subplot(gs[g[2]:g[3]+1], projection=_proj(n))
        else:
            ax = fig.add_axes(n["pos"], projection=_proj(n))
        made[n["i"]] = ax
    for n in nodes:
        if not is_twin(n): continue
        base = made.get(min(n["twin_group"]))
        if base is None: continue
        # twinx shares x; twiny shares y
        shares_x = min(n["twin_group"]) in n["sharex"]
        made[n["i"]] = base.twinx() if shares_x else base.twiny()
        made[n["i"]].set_position(n["pos"])
    for n in nodes:
        ax = made.get(n["i"])
        if ax is None: continue
        try: ax.set_position(n["pos"])      # exact saved layout
        except Exception: pass
        ax.set_facecolor(n["facecolor"])
        for loc, tt in (n.get("titles") or {"center": {"text": n["title"], "style": {}}}).items():
            ax.set_title(tt["text"], loc=loc)
            _apply_text_style({"center": ax.title, "left": ax._left_title,
                               "right": ax._right_title}[loc], tt.get("style"))
        try: ax.set_aspect(n["aspect"])
        except Exception: pass
        for cd in n.get("containers", []):
            try:
                _replay_container(ax, cd, arrays, report)
            except Exception as e:
                report.append(f"container {cd.get('cls')}: {type(e).__name__}: {e}")
        for an in n["artists"]:
            art = _make_artist(an, ax, arrays, report)
            if art is None: continue
            if isinstance(art, tuple) and art[0] == "attached":
                art = art[1]
            else:
                _apply_props(art, an.get("props", {}), arrays, report)
                _apply_payload(art, an, arrays, report)
                if isinstance(art, Line2D): ax.add_line(art)
                elif isinstance(art, Collection): ax.add_collection(art)
                elif isinstance(art, Patch): ax.add_patch(art)
                elif isinstance(art, Text): ax.add_artist(art)
                else: ax.add_artist(art)
            tf = an.get("tf")
            if tf == "xaxis": art.set_transform(ax.get_xaxis_transform())
            elif tf == "yaxis": art.set_transform(ax.get_yaxis_transform())
            elif tf == "axes": art.set_transform(ax.transAxes)
        for k, s in n.get("spines", {}).items():
            if k in ax.spines:
                ax.spines[k].set_visible(s["visible"])
                ax.spines[k].set_color(s["color"]); ax.spines[k].set_linewidth(s["lw"])
        _apply_axis(ax, "x", n["x"], report)
        _apply_axis(ax, "y", n["y"], report)
        if n.get("legend"):
            h, l = ax.get_legend_handles_labels()
            if l:
                lg = n["legend"]
                kw = dict(title=lg["title"] or None, frameon=lg["frameon"])
                ts = lg.get("text_style") or {}
                if ts.get("size") is not None: kw["fontsize"] = ts["size"]
                if (lg.get("title_style") or {}).get("size") is not None:
                    kw["title_fontsize"] = lg["title_style"]["size"]
                try:
                    leg_obj = ax.legend(loc=lg.get("loc", "best"),
                                        ncols=lg.get("ncols", 1), **kw)
                except Exception:
                    leg_obj = ax.legend(**kw)
                for t_ in leg_obj.get_texts(): _apply_text_style(t_, ts)
                _apply_text_style(leg_obj.get_title(), lg.get("title_style"))
    # Colorbars: the cbar axes was created at its saved position like any other
    # axes, so attach with cax= — using ax= would shrink the parent a 2nd time.
    for ci, info in (tree.get("colorbars") or {}).items():
        try:
            cax = made.get(int(ci))
            parent = made.get(int(info.get("parent")))
            if cax is None or parent is None: continue
            mappable = (list(parent.images) or list(parent.collections) or [None])[0]
            if mappable is None: continue
            cax.clear()
            cb = fig.colorbar(mappable, cax=cax,
                              orientation=info.get("orientation", "vertical"))
            if info.get("label"): cb.set_label(info["label"])
        except Exception as e:
            report.append(f"colorbar: {type(e).__name__}: {e}")

    # final pass: restore autoscale MODE after every limit has been set
    for n in nodes:
        ax = made.get(n["i"])
        if ax is None: continue
        ax.set_autoscalex_on(bool(n["x"].get("autoscale")))
        ax.set_autoscaley_on(bool(n["y"].get("autoscale")))
    if f.get("suptitle"):
        st = fig.suptitle(f["suptitle"])
        _apply_text_style(st, f.get("suptitle_style"))
    fig._mfs_scale = float(f.get("mfs_scale", 1.0))
    if figsize is not None:
        resize_figure(fig, figsize, scale)    # relation to the SAVED size
    tree.setdefault("replay_report", []).extend(report)
    return fig

# -------------------------------------------------------------------- file io
def save_scene(fig, path):
    tree, arrays = dump_figure(fig)
    buf = io.BytesIO(); np.savez_compressed(buf, **arrays.d)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("tree.json", json.dumps(tree, indent=1))
        z.writestr("arrays.npz", buf.getvalue())
    return path, tree

def load_scene(path):
    with zipfile.ZipFile(path) as z:
        tree = json.loads(z.read("tree.json"))
        npz = np.load(io.BytesIO(z.read("arrays.npz")), allow_pickle=False)
        arrays = _Arrays(); arrays.d = {k: npz[k] for k in npz.files}
    return tree, arrays


# ===========================================================================
#  DATA ACCESS LAYER
#  MATLAB's .fig has no data API at all — you dig through handle graphics.
#  Here the scene tree already holds every array, so we expose a queryable,
#  exportable view over it.
# ===========================================================================

class Series:
    """One plotted thing, with uniform ``x``/``y`` plus kind-specific extras.

    Always present: ``kind``, ``label``, ``x``, ``y``, ``axes`` (index).
    Extras by kind:
      fill      -> ``lo``, ``hi`` (y is lo), ``orient``
      errorbar  -> ``xerr``, ``yerr`` (each (n,2) lo/hi magnitudes, or None)
      scatter   -> ``sizes``, ``colors``
      bar       -> ``widths``, ``bottoms``, ``orient``
      image/mesh-> ``z`` (2-D), ``extent`` or ``xy``

    Unpacks like a tuple:  ``x, y = series``
    """
    __slots__ = ("kind", "label", "x", "y", "axes", "extra")

    def __init__(self, kind, label, x, y, axes, **extra):
        self.kind, self.label, self.x, self.y, self.axes = kind, label, x, y, axes
        self.extra = extra

    def __getattr__(self, k):
        try: return self.extra[k]
        except KeyError:
            raise AttributeError(f"{self.kind} series has no {k!r}") from None

    def __iter__(self):
        yield self.x; yield self.y

    def __len__(self):
        return 0 if self.x is None else len(np.atleast_1d(self.x))

    def keys(self):
        """Column names, like a dict / DataFrame."""
        return list(self.columns().keys())

    def __getitem__(self, k):
        """series['x'] / series['y'] — dict-style access."""
        c = self.columns()
        if k in c: return c[k]
        if k in self.extra: return self.extra[k]
        raise KeyError(k)

    def __contains__(self, k): return k in self.columns() or k in self.extra

    def to_dict(self):
        """Plain {name: array} dict — the least surprising export."""
        return dict(self.columns())

    def columns(self):
        """Named 1-D columns for CSV/DataFrame export."""
        out = {}
        if self.x is not None and np.ndim(self.x) == 1: out["x"] = self.x
        if self.y is not None and np.ndim(self.y) == 1: out["y"] = self.y
        for k in ("lo", "hi", "sizes", "widths", "bottoms"):
            v = self.extra.get(k)
            if v is not None and np.ndim(v) == 1: out[k] = v
        for k in ("xerr", "yerr"):
            v = self.extra.get(k)
            if v is not None and np.ndim(v) == 2 and v.shape[1] == 2:
                out[k + "_lo"], out[k + "_hi"] = v[:, 0], v[:, 1]
        return out

    def to_dataframe(self):
        """pandas DataFrame of this series' columns (needs pandas installed)."""
        import pandas as pd
        return pd.DataFrame(self.columns())

    def plot(self, ax=None, **kw):
        """Re-draw just this series onto an Axes."""
        ax = ax or plt.gca()
        lbl = kw.pop("label", self.label or None)
        if self.kind == "fill":
            f = ax.fill_betweenx if self.extra.get("orient") == "h" else ax.fill_between
            return f(self.x, self.extra["lo"], self.extra["hi"], label=lbl, **kw)
        if self.kind == "scatter":
            return ax.scatter(self.x, self.y, s=self.extra.get("sizes"), label=lbl, **kw)
        if self.kind == "errorbar":
            xe, ye = self.extra.get("xerr"), self.extra.get("yerr")
            return ax.errorbar(self.x, self.y, label=lbl,
                               xerr=None if xe is None else xe.T,
                               yerr=None if ye is None else ye.T,
                               fmt=kw.pop("fmt", "o"), **kw)
        if self.kind == "bar":
            return ax.bar(self.x, self.y, width=self.extra.get("widths"),
                          bottom=self.extra.get("bottoms"), align="edge",
                          label=lbl, **kw)
        if self.kind in ("image", "mesh"):
            return ax.imshow(self.extra["z"], extent=self.extra.get("extent"), **kw)
        if self.kind == "vline":  return ax.axvline(float(np.ravel(self.x)[0]), **kw)
        if self.kind == "hline":  return ax.axhline(float(np.ravel(self.y)[0]), **kw)
        return ax.plot(self.x, self.y, label=lbl, **kw)

    def __repr__(self):
        n = len(self)
        return f"<Series {self.kind} {self.label!r} n={n} ax={self.axes}>"


class _Group(list):
    """A list of Series that can also be indexed by label or filtered by kind."""

    def __getitem__(self, k):
        if isinstance(k, str):
            hits = [s for s in self if s.label == k]
            if not hits:
                raise KeyError(f"no series labelled {k!r}; have "
                               f"{[s.label for s in self if s.label]}")
            return hits[0] if len(hits) == 1 else _Group(hits)
        return list.__getitem__(self, k)

    @property
    def labels(self): return [s.label for s in self if s.label]

    def keys(self):
        """Labels, like dict.keys()."""
        return self.labels

    def values(self): return list(self)

    def items(self):
        """(label, series) pairs, like dict.items()."""
        return [(s.label or f"{s.kind}{i}", s) for i, s in enumerate(self)]

    def get(self, label, default=None):
        try: return self[label]
        except KeyError: return default

    def __contains__(self, k):
        return any(s.label == k for s in self) if isinstance(k, str) else list.__contains__(self, k)

    def of_kind(self, kind): return _Group(s for s in self if s.kind == kind)

    def to_dataframe(self):
        """Long-form DataFrame of every series (needs pandas)."""
        import pandas as pd
        frames = []
        for i, s in enumerate(self):
            df = pd.DataFrame(s.columns())
            if df.empty: continue
            df.insert(0, "label", s.label or f"{s.kind}{i}")
            df.insert(0, "kind", s.kind); df.insert(0, "axes", s.axes)
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def to_csv(self, path):
        """Write every series to one long-form CSV (no pandas required)."""
        import csv
        rows, cols = [], []
        for i, s in enumerate(self):
            c = s.columns()
            for k in c:
                if k not in cols: cols.append(k)
            n = max((len(v) for v in c.values()), default=0)
            for j in range(n):
                r = {"axes": s.axes, "kind": s.kind,
                     "label": s.label or f"{s.kind}{i}"}
                r.update({k: (v[j] if j < len(v) else "") for k, v in c.items()})
                rows.append(r)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["axes", "kind", "label"] + cols)
            w.writeheader(); w.writerows(rows)
        return path

    def __repr__(self):
        if not self: return "<no series>"
        w = max(len(s.kind) for s in self)
        body = "\n".join(f"  [{i}] {s.kind:<{w}}  {s.label or '-':<14} n={len(s)}"
                         for i, s in enumerate(self))
        return f"<{len(self)} series>\n{body}"


class FigureData:
    """Queryable data view of a saved figure.

    >>> d = read_data("run.mplscene")
    >>> d.labels                 # what's in here
    >>> x, y = d["signal"]       # by label, unpacks to arrays
    >>> d.axes[1].of_kind("fill")[0].hi
    >>> d.to_csv("run.csv")
    """

    def __init__(self, axes_groups, meta=None):
        self.axes = axes_groups                    # list[_Group]
        self.all = _Group(s for g in axes_groups for s in g)
        self.meta = meta or {}

    # --- dict protocol: behaves like {label: series} ---
    def __getitem__(self, k): return self.all[k]
    def __iter__(self): return iter(self.all)
    def __len__(self): return len(self.all)
    def __contains__(self, k): return k in self.all
    def keys(self): return self.all.keys()
    def values(self): return self.all.values()
    def items(self): return self.all.items()
    def get(self, k, default=None): return self.all.get(k, default)

    @property
    def labels(self): return self.all.labels
    def of_kind(self, kind): return self.all.of_kind(kind)
    def to_dataframe(self): return self.all.to_dataframe()
    def to_csv(self, path): return self.all.to_csv(path)

    def to_dict(self):
        """{label: {column: array}} — plain nested dict, no custom classes."""
        return {lbl: s.to_dict() for lbl, s in self.items()}

    def replot(self, **kw):
        """Quick figure with every series re-drawn on its original axes index."""
        n = max(1, len(self.axes))
        fig, axs = plt.subplots(n, 1, squeeze=False, **kw)
        for g, ax in zip(self.axes, axs.ravel()):
            for s in g: s.plot(ax)
            if g.labels: ax.legend()
        return fig

    def __repr__(self):
        head = f"FigureData: {len(self.axes)} axes, {len(self.all)} series"
        return head + ("\n" + repr(self.all) if self.all else "")


def extract_data(tree, arrays):
    """Build a FigureData view from a dumped scene tree."""
    groups = []
    for ai, node in enumerate(tree.get("axes", [])):
        g = _Group()
        for cd in node.get("containers", []):
            if cd.get("cls") == "ErrorbarContainer":
                err = cd.get("err") or {}
                g.append(Series("errorbar", cd.get("label", ""),
                                arrays.get(cd["x"]), arrays.get(cd["y"]), ai,
                                xerr=arrays.get(err["x"]) if err.get("x") else None,
                                yerr=arrays.get(err["y"]) if err.get("y") else None))
            elif cd.get("cls") == "BarContainer":
                g.append(Series("bar", cd.get("label", ""),
                                arrays.get(cd["x"]), arrays.get(cd["values"]), ai,
                                widths=arrays.get(cd["widths"]),
                                bottoms=arrays.get(cd["bottoms"]) if cd.get("bottoms") else None,
                                orient="h" if cd.get("orientation") == "horizontal" else "v"))
        for an in node.get("artists", []):
            base, cls = an.get("base"), an.get("cls")
            lbl = an.get("label", "")
            lbl = "" if isinstance(lbl, str) and lbl.startswith("_") else lbl
            if base == "line":
                x = arrays.get(an["data"][0]); y = arrays.get(an["data"][1])
                tf = an.get("tf")
                if tf == "xaxis":   g.append(Series("vline", lbl, x, None, ai))
                elif tf == "yaxis": g.append(Series("hline", lbl, None, y, ai))
                else:               g.append(Series("line", lbl, x, y, ai))
            elif base == "image":
                z = arrays.get(an["array"])
                g.append(Series("image", lbl, None, None, ai,
                                z=z, extent=an.get("extent")))
            elif base == "quadmesh":
                c = arrays.get(an["coords"]); z = arrays.get(an["array"])
                g.append(Series("mesh", lbl, c[..., 0], c[..., 1], ai, z=z))
            elif base == "collection":
                if "offsets" in an and "sizes" in an:
                    off = arrays.get(an["offsets"])
                    g.append(Series("scatter", lbl, off[:, 0], off[:, 1], ai,
                                    sizes=arrays.get(an["sizes"]),
                                    colors=_dec(an.get("fc"), arrays)))
                else:
                    fb = _fill_from_paths(an, arrays)
                    if fb: g.append(Series("fill", lbl, fb["t"], fb["lo"], ai,
                                           lo=fb["lo"], hi=fb["hi"], orient=fb["orient"]))
            elif base == "text":
                g.append(Series("text", an.get("text", ""),
                                np.array([an["pos"][0]]), np.array([an["pos"][1]]), ai))
        groups.append(g)
    return FigureData(groups, meta={k: tree.get(k) for k in ("mpl", "np", "schema")})


def _fill_from_paths(node, arrays):
    """Recover (t, lo, hi) for a filled band from its polygon path(s)."""
    orient = "v" if node.get("cls") != "FillBetweenPolyCollectionH" else "h"
    ts, los, his = [], [], []
    for pd_ in node.get("paths", []):
        p = _dec(pd_, arrays)
        v = p.vertices
        if p.codes is not None: v = v[p.codes != MPath.CLOSEPOLY]
        if len(v) < 4: continue
        n = (len(v) - 2) // 2
        lower, upper = v[1:1 + n], v[1 + n + 1:1 + n + 1 + n][::-1]
        if len(upper) != n: continue
        if ts: ts.append([np.nan]); los.append([np.nan]); his.append([np.nan])
        ts.append(lower[:, 0]); los.append(lower[:, 1]); his.append(upper[:, 1])
    if not ts: return None
    return {"orient": orient, "t": np.concatenate(ts),
            "lo": np.concatenate(los), "hi": np.concatenate(his)}


def figure_data(fig):
    """FigureData straight from a live Figure."""
    tree, arrays = dump_figure(fig)
    return extract_data(tree, arrays)


def read_data(path):
    """FigureData from a saved .mplscene file."""
    tree, arrays = load_scene(path)
    return extract_data(tree, arrays)


# ===========================================================================
#  SIZE-AWARE SCALING
#  matplotlib keeps figsize (inches) and every size (points) independent, which
#  is right for publication work (8 pt text must stay 8 pt). But when a figure
#  is *resized*, everything should scale together or it looks wrong. These are
#  explicit, opt-in tools: nothing here runs unless you call it.
# ===========================================================================

def _scale_axes(ax, k):
    for ln in ax.lines:
        ln.set_linewidth(ln.get_linewidth() * k)
        ln.set_markersize(ln.get_markersize() * k)
        ln.set_markeredgewidth(ln.get_markeredgewidth() * k)
    for c in ax.collections:
        try:
            lw = np.asarray(c.get_linewidth(), float)
            if lw.size: c.set_linewidth(lw * k)
            sz = np.asarray(c.get_sizes(), float)
            if sz.size: c.set_sizes(sz * k * k)      # marker AREA scales as k^2
        except Exception:
            pass
    for p in ax.patches:
        p.set_linewidth(p.get_linewidth() * k)
        if isinstance(p, FancyArrowPatch):       # e.g. streamplot arrowheads
            p.set_mutation_scale(p.get_mutation_scale() * k)
    for t in ax.texts:
        ap = getattr(t, "arrow_patch", None)
        if ap is not None:
            ap.set_linewidth(ap.get_linewidth() * k)
            ap.set_mutation_scale(ap.get_mutation_scale() * k)
    for sp in ax.spines.values():
        sp.set_linewidth(sp.get_linewidth() * k)
    for axis in (ax.xaxis, ax.yaxis):
        axis.labelpad = axis.labelpad * k
        for which, ticks in (("major", axis.majorTicks), ("minor", axis.minorTicks)):
            if not ticks: continue
            t0 = ticks[0]
            axis.set_tick_params(
                which=which,
                length=t0.tick1line.get_markersize() * k,
                width=t0.tick1line.get_markeredgewidth() * k,
                pad=t0.get_pad() * k,
                labelsize=t0.label1.get_fontsize() * k,
                grid_linewidth=t0.gridline.get_linewidth() * k)
    leg = ax.get_legend()
    if leg is not None:
        try: leg.get_frame().set_linewidth(leg.get_frame().get_linewidth() * k)
        except Exception: pass
        handles = (getattr(leg, "legend_handles", None)
                   or getattr(leg, "legendHandles", []) or [])
        for h in handles:
            try:
                if hasattr(h, "set_linewidth"):
                    lw = np.asarray(h.get_linewidth(), float)
                    h.set_linewidth(lw * k if lw.ndim else float(lw) * k)
                if hasattr(h, "set_markersize"):
                    h.set_markersize(h.get_markersize() * k)
                if hasattr(h, "get_sizes") and np.asarray(h.get_sizes()).size:
                    h.set_sizes(np.asarray(h.get_sizes(), float) * k * k)
            except Exception:
                pass


def scale_figure(fig, factor=None, *, reference=None):
    """Scale every size in a figure together: fonts, line widths, markers, ticks.

    Does NOT change the figure's size in inches; see :func:`resize_figure` for
    that. matplotlib keeps figsize and point sizes independent on purpose (so
    that 8 pt text stays 8 pt), so nothing is scaled unless you call this.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    factor : float | None
        Multiply all sizes by this. ``None`` means "fit the figure's current
        size", i.e. ``min(width/ref_w, height/ref_h)`` relative to *reference*.
        The ``None`` form is idempotent: calling it twice does nothing the
        second time (the applied scale is remembered on the figure).
    reference : (float, float) | None
        The (width, height) in inches at which the current sizes look right.
        Defaults to matplotlib's design size, ``rcParamsDefault['figure.figsize']``
        (6.4 x 4.8).

    Returns
    -------
    Figure
        The same figure, for chaining.

    Examples
    --------
    >>> fig, ax = plt.subplots(figsize=(12, 8))   # fonts look tiny at 12x8
    >>> mfs.scale_figure(fig)                      # fit them to the size
    >>> mfs.scale_figure(fig, 1.5)                 # or by an explicit factor
    """
    cur = float(getattr(fig, "_mfs_scale", 1.0))
    if factor is None:
        ref = reference or matplotlib.rcParamsDefault["figure.figsize"]
        w, h = fig.get_size_inches()
        k = min(w / ref[0], h / ref[1]) / cur
    else:
        k = float(factor)
    if not np.isfinite(k) or k <= 0:
        raise ValueError(f"scale factor must be a positive number, got {k!r}")
    if abs(k - 1.0) < 1e-12:
        return fig

    skip = set()                       # tick labels are scaled via tick params
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for t in list(axis.majorTicks) + list(axis.minorTicks):
                skip.update((id(t.label1), id(t.label2)))
            skip.add(id(axis.offsetText))
    for t in fig.findobj(Text):
        if id(t) not in skip:
            t.set_fontsize(t.get_fontsize() * k)
    for ax in fig.axes:
        try: _scale_axes(ax, k)
        except Exception: pass
    fig._mfs_scale = cur * k
    return fig


def resize_figure(fig, figsize, scale="auto"):
    """Change a figure's size in inches, scaling its contents to match.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    figsize : (float, float)
        New (width, height) in inches.
    scale : "auto" | float | False
        ``"auto"``: scale by ``min(new_w/old_w, new_h/old_h)`` so the figure
        keeps its proportions and nothing overflows. A number forces that
        factor. ``False`` resizes without touching fonts (plain
        ``set_size_inches``).
    """
    w0, h0 = fig.get_size_inches()
    w1, h1 = float(figsize[0]), float(figsize[1])
    if scale == "auto":
        k = min(w1 / w0, h1 / h0)
    elif scale is False or scale is None:
        k = 1.0
    else:
        k = float(scale)
    fig.set_size_inches(w1, h1)
    if k != 1.0:
        scale_figure(fig, k)
    return fig
