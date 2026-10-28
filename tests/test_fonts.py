"""Font/text-style round-trip. Compares PROPERTY VALUES, not pixels: text is a
tiny share of an image, so pixel diffs cannot see these regressions."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, warnings
warnings.simplefilter('ignore')
import mpl_scene as S

def readout(fig):
    ax = fig.axes[0]; leg = ax.get_legend()
    return {
        'title_left': (ax.get_title(loc='left'), ax._left_title.get_fontsize(), ax._left_title.get_fontweight()),
        'title_right': (ax.get_title(loc='right'), ax._right_title.get_fontsize()),
        'xlabel': (ax.get_xlabel(), ax.xaxis.label.get_fontsize(), ax.xaxis.label.get_fontstyle()),
        'ylabel': (ax.get_ylabel(), ax.yaxis.label.get_fontsize()),
        'xtick': ax.get_xticklabels()[0].get_fontsize(),
        'ytick': ax.get_yticklabels()[0].get_fontsize(),
        'legend': leg.get_texts()[0].get_fontsize(),
        'legend_title': (leg.get_title().get_text(), leg.get_title().get_fontsize()),
        'suptitle': (fig._suptitle.get_text(), fig._suptitle.get_fontsize(), fig._suptitle.get_fontweight()),
    }

x = np.linspace(0, 6, 50)
f, a = plt.subplots(figsize=(7, 4.5))
a.plot(x, np.sin(x), label='sig'); a.plot(x, np.cos(x), label='cos')
a.set_title('Left', fontsize=20, fontweight='bold', loc='left')
a.set_title('Right', fontsize=11, loc='right')
a.set_xlabel('time', fontsize=15, style='italic'); a.set_ylabel('amp', fontsize=9)
a.tick_params(axis='x', labelsize=13); a.tick_params(axis='y', labelsize=7)
a.legend(fontsize=12, title='trig', title_fontsize=14)
f.suptitle('Super', fontsize=22, fontweight='light')

want = readout(f)
tree, arr = S.dump_figure(f)
# replay under DIFFERENT rcParams, as a user with another style would
with plt.rc_context({'font.size': 6, 'axes.titlesize': 6, 'axes.labelsize': 6,
                     'legend.fontsize': 6, 'figure.titlesize': 6}):
    got = readout(S.replay_figure(tree, arr))

bad = [k for k in want if want[k] != got[k]]
for k in want: print(f"{'OK ' if k not in bad else 'BAD'} {k:13s} {want[k]}  ->  {got[k]}")
print("\nRESULT:", "ALL PASS" if not bad else f"{len(bad)} FAIL {bad}")
