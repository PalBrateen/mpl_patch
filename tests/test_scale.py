"""Scale-invariance oracle. A figure at size S / dpi D must render the same as
the same figure scaled by k at size k*S / dpi D/k: every element then has an
identical PIXEL size. Any residual difference marks something left unscaled."""
import matplotlib; matplotlib.use('Agg')
import numpy as np, io, sys, warnings
warnings.simplefilter('ignore')
import matplotlib.image as mi
import battery25 as B, mpl_scene as S

def render(fig, dpi):
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=dpi); b.seek(0)
    return mi.imread(b)[..., :3]

def run(k, verbose=True):
    base, scaled = B.build(), B.build()
    out = {}
    for name in base:
        f0, f1 = base[name], scaled[name]
        w, h = f0.get_size_inches()
        S.resize_figure(f1, (w * k, h * k))
        a, b = render(f0, 100), render(f1, 100 / k)
        out[name] = float(np.abs(a - b).mean()) if a.shape == b.shape else float('nan')
    return out

if __name__ == "__main__":
    for k in (2.0, 0.5):
        r = run(k)
        worst = sorted(r.items(), key=lambda kv: -np.nan_to_num(kv[1], nan=9))[:8]
        print(f"\nk={k}: median {np.median(list(r.values())):.4f}  max {np.nanmax(list(r.values())):.4f}")
        for n, d in worst: print(f"   {n:16s} {d:.4f}")
