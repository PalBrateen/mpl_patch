import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, io, json, warnings
warnings.simplefilter('ignore')
import mpl_scene as S

def figs():
    r=np.random.default_rng(0); x=np.linspace(0,2*np.pi,80)
    F={}
    f,a=plt.subplots(); a.plot(x,np.sin(x),label='s'); a.plot(x,np.cos(x)); a.legend(); F['lines']=f
    f,a=plt.subplots(); a.plot(x,np.sin(x)); a.fill_between(x,np.sin(x)-.2,np.sin(x)+.2,alpha=.3); F['fill']=f
    f,a=plt.subplots(); a.scatter(r.normal(size=60),r.normal(size=60),s=r.uniform(5,80,60),alpha=.6); F['scatter']=f
    f,a=plt.subplots(); a.errorbar(x[::8],np.sin(x[::8]),yerr=.1,fmt='o'); F['errorbar']=f
    f,a=plt.subplots(); a.bar(np.arange(6),r.uniform(1,5,6)); F['bar']=f
    f,a=plt.subplots(); a.imshow(r.random((12,15)),extent=[0,1,0,1],cmap='viridis'); F['imshow']=f
    f,a=plt.subplots(); a.pcolormesh(np.arange(9),np.arange(7),r.random((6,8)),cmap='magma'); F['pcolormesh']=f
    f,a=plt.subplots(); a.contourf(r.random((10,10))); F['contourf']=f
    f,a=plt.subplots(); a.plot(x,np.exp(x/3)); a.set_yscale('log'); F['logscale']=f
    f,a=plt.subplots(); a.plot(x,np.sin(x)); b=a.twinx(); b.plot(x,np.exp(-x),'r'); a.set_ylabel('L',color='b'); b.set_ylabel('R',color='r'); F['twinx']=f
    f,axs=plt.subplots(2,2); [ax.plot(x,np.sin(x+k)) for k,ax in enumerate(axs.ravel())]; F['grid22']=f
    f,a=plt.subplots(); a.plot(x,np.sin(x)); a.axvline(3,color='k',ls='--'); a.axhline(0,color='g'); a.text(1,.5,'hi'); F['refs_text']=f
    f,a=plt.subplots(); a.stackplot(np.arange(10),r.random((3,10))); F['stackplot']=f
    f,a=plt.subplots(); a.plot(x,np.sin(x)); a.grid(True); a.minorticks_on(); F['grid_minor']=f
    return F

def canon(o, arrays):
    """Replace array REFERENCES by a hash of their CONTENT, so the comparison
    tests the data rather than the arbitrary a0/a1 naming."""
    import hashlib
    if isinstance(o, dict):
        if "__arr__" in o:
            a = np.asarray(arrays.d[o["__arr__"]])
            return "arr:" + hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:12] + f":{a.shape}"
        return {k: canon(v, arrays) for k, v in o.items()}
    if isinstance(o, list): return [canon(v, arrays) for v in o]
    return o

def norm(tree, arrays):
    t=json.loads(json.dumps(tree))
    for k in ('report','replay_report','mpl','np'): t.pop(k,None)
    return canon(t, arrays)

def px(fig):
    b=io.BytesIO(); fig.savefig(b,format='png',dpi=50); b.seek(0)
    import matplotlib.image as mi; return mi.imread(b)

print(f"{'figure':12s} {'fixpoint':9s} {'pixdiff':>8s}  notes")
nfail=0
for name,f in figs().items():
    t1,a1=S.dump_figure(f)
    f2=S.replay_figure(t1,a1)
    t2,a2=S.dump_figure(f2)
    same = norm(t1,a1)==norm(t2,a2)
    try:
        p1,p2=px(f),px(f2)
        d=float(np.abs(p1[...,:3]-p2[...,:3]).mean()) if p1.shape==p2.shape else float('nan')
    except Exception as e: d=float('nan')
    notes=[]
    rep=t1.get('report',[])+t1.get('replay_report',[])
    if rep:
        from collections import Counter
        c=Counter(x.split(':')[0].split(' @')[0] for x in rep)
        notes=[f"{k}x{v}" for k,v in c.most_common(3)]
    if not same: nfail+=1
    print(f"{name:12s} {'OK' if same else 'DIFF':9s} {d:8.4f}  {'; '.join(notes)[:60]}")
print("\nfixpoint failures:",nfail)
