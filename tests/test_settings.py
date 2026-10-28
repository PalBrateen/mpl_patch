import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, tempfile, os, warnings, pickle
warnings.simplefilter('ignore')
import mpl_figure_saver as mfs
fails=[]
def ok(c,msg): print(("PASS" if c else "FAIL"),msg); (fails.append(msg) if not c else None)
def raises(exc,fn):
    try: fn()
    except exc as e: return str(e)
    except Exception as e: return None
    return None

mfs.install(save_dir=tempfile.mkdtemp())
S=mfs.settings
ok(S.save_pickle is True and S.save_data is True,"defaults on")

# typo is LOUD, with a suggestion  (the whole reason for not using bare module attrs)
msg=raises(AttributeError,lambda: setattr(S,'save_picke',False))
ok(msg is not None and "save_pickle" in msg,f"typo raises + suggests: {msg[:70] if msg else msg}")
ok(raises(AttributeError,lambda: S.nonexistent) is not None,"unknown read raises")

# "False" (a string) is truthy -> must be rejected, not silently accepted
ok(raises(TypeError,lambda: setattr(S,'save_data',"False")) is not None,"string 'False' rejected")
ok(raises(TypeError,lambda: setattr(S,'save_data',0)) is not None,"int 0 rejected (must be a real bool)")
ok(raises(ValueError,lambda: setattr(S,'widget_mode','dialgo')) is not None,"bad widget_mode rejected")

# behaviour: switching off really changes what is written
td=tempfile.mkdtemp(); f,a=plt.subplots(); a.plot([0,1],[0,1])
S.save_pickle=False; f.savefig(os.path.join(td,'a.png'))
ok(sorted(os.listdir(td))==['a.mplscene','a.png'],f"pickle off -> {sorted(os.listdir(td))}")
S.save_data=False; f.savefig(os.path.join(td,'b.png'))
ok([x for x in os.listdir(td) if x.startswith('b')]==['b.png'],"both off -> image only")
S.reset()

# context manager: temporary, and restored even when the body raises
td=tempfile.mkdtemp()
with S(save_pickle=False):
    ok(S.save_pickle is False,"inside with: off")
    f.savefig(os.path.join(td,'c.png'))
ok(S.save_pickle is True,"after with: restored")
ok(sorted(os.listdir(td))==['c.mplscene','c.png'],"with-block save wrote no pickle")
try:
    with S(save_data=False): raise RuntimeError("boom")
except RuntimeError: pass
ok(S.save_data is True,"restored after exception")

# call form is a permanent switch too (torch.set_grad_enabled-style)
S(save_pickle=False); ok(S.save_pickle is False,"bare call applies permanently"); S.reset()

# all-or-nothing: a bad value in the middle must not leave a half-applied state
before=S.as_dict()
raises(TypeError,lambda: S(save_data=False, save_pickle="no"))
ok(S.as_dict()==before,"failed multi-set rolls back completely")

# save_dir: resolved to ABSOLUTE at assignment time, created, immune to chdir
rel=tempfile.mkdtemp(); cwd=os.getcwd(); os.chdir(rel)
S.save_dir='figs_here'; resolved=S.save_dir; os.chdir(cwd)
ok(resolved.is_absolute() and resolved.exists() and str(resolved).startswith(os.path.realpath(rel)),"save_dir absolute+created")
ok(S.save_dir==resolved,"save_dir unaffected by later chdir")

# back-compat shims forward to the same object
mfs.configure(save_data=False); ok(S.save_data is False,"configure() forwards"); S.reset()
p=mfs.set_save_dir(tempfile.mkdtemp()); ok(S.save_dir==p,"set_save_dir() forwards")
mfs.install(widget_mode='dialog',save_pickle=False)
ok(S.widget_mode=='dialog' and S.save_pickle is False,"install(kwargs) forwards"); S.reset()

# repr + autocomplete surface
ok('save_pickle' in repr(S) and 'save_pickle' in dir(S),"repr/dir list the settings")
print("\nRESULT:","ALL PASS" if not fails else f"{len(fails)} FAIL -> {fails}")
