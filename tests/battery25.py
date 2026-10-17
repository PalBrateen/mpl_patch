import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, io, json, warnings, hashlib
warnings.simplefilter('ignore')
import matplotlib.image as mi
import mpl_scene as S

def R(): return np.random.default_rng(0)
def build():
    x=np.linspace(0,2*np.pi,80); r=R(); F={}
    def mk(name):
        def deco(fn):
            f=plt.figure(); fn(f); F[name]=f
        return deco
    @mk('hist')
    def _(f): f.add_subplot().hist(r.normal(size=500),bins=20)
    @mk('step')
    def _(f): f.add_subplot().step(np.arange(10),r.random(10),where='mid')
    @mk('stem')
    def _(f): f.add_subplot().stem(np.arange(10),r.random(10))
    @mk('boxplot')
    def _(f): f.add_subplot().boxplot([r.normal(size=50),r.normal(size=50)])
    @mk('violin')
    def _(f): f.add_subplot().violinplot([r.normal(size=50),r.normal(size=50)])
    @mk('quiver')
    def _(f):
        a=f.add_subplot(); X,Y=np.meshgrid(np.arange(5),np.arange(5)); a.quiver(X,Y,X*.1,Y*.1)
    @mk('streamplot')
    def _(f):
        a=f.add_subplot(); Y,X=np.mgrid[-2:2:20j,-2:2:20j]; a.streamplot(X,Y,-Y,X)
    @mk('contour_lines')
    def _(f): f.add_subplot().contour(r.random((10,10)))
    @mk('colorbar')
    def _(f):
        a=f.add_subplot(); im=a.imshow(r.random((8,8))); f.colorbar(im,ax=a)
    @mk('polar')
    def _(f):
        a=f.add_subplot(projection='polar'); a.plot(np.linspace(0,2*np.pi,50),np.linspace(0,1,50))
    @mk('annotate')
    def _(f):
        a=f.add_subplot(); a.plot(x,np.sin(x))
        a.annotate('peak',xy=(np.pi/2,1),xytext=(2,0.5),arrowprops=dict(arrowstyle='->'))
    @mk('dates')
    def _(f):
        import datetime
        a=f.add_subplot(); d=[datetime.datetime(2026,1,1)+datetime.timedelta(days=i) for i in range(20)]
        a.plot(d,r.random(20)); f.autofmt_xdate()
    @mk('loglog')
    def _(f): f.add_subplot().loglog(np.logspace(0,3,50),np.logspace(0,3,50)**1.5)
    @mk('sharex_sub')
    def _(f):
        axs=f.subplots(2,1,sharex=True); axs[0].plot(x,np.sin(x)); axs[1].plot(x,np.cos(x))
    @mk('hlines_coll')
    def _(f): f.add_subplot().hlines([1,2,3],0,1,colors=['r','g','b'])
    @mk('fill_where')
    def _(f):
        a=f.add_subplot(); a.fill_between(x,0,np.sin(x),where=np.sin(x)>0,alpha=.4)
    @mk('text_styles')
    def _(f):
        a=f.add_subplot(); a.plot(x,np.sin(x))
        a.text(1,.5,'big',fontsize=18,color='purple',rotation=15,ha='center')
    @mk('spines_off')
    def _(f):
        a=f.add_subplot(); a.plot(x,np.sin(x))
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
        a.spines['left'].set_color('red'); a.spines['left'].set_linewidth(3)
    @mk('grid_styled')
    def _(f):
        a=f.add_subplot(); a.plot(x,np.sin(x)); a.grid(True,ls=':',color='gray',alpha=.7)
    @mk('legend_outside')
    def _(f):
        a=f.add_subplot(); a.plot(x,np.sin(x),label='s'); a.plot(x,np.cos(x),label='c')
        a.legend(loc='upper right',ncols=2,title='trig',frameon=True)
    @mk('twin_both')
    def _(f):
        a=f.add_subplot(); a.plot(x,np.sin(x)); b=a.twinx(); b.plot(x,np.exp(-x),'r')
        c=a.twiny(); c.plot(np.linspace(0,10,80),np.cos(x),'g')
    @mk('axhspan')
    def _(f):
        a=f.add_subplot(); a.plot(x,np.sin(x)); a.axhspan(-.5,.5,alpha=.2,color='y'); a.axvspan(1,2,alpha=.2,color='c')
    @mk('masked')
    def _(f):
        a=f.add_subplot(); y=np.ma.masked_where(np.sin(x)<0,np.sin(x)); a.plot(x,y)
    @mk('markers')
    def _(f):
        a=f.add_subplot()
        for i,m in enumerate(['o','^','s','*','D']):
            a.plot(x,np.sin(x)+i*.3,marker=m,markevery=8,ls='--',ms=8)
    @mk('scatter_cmap')
    def _(f):
        a=f.add_subplot(); sc=a.scatter(r.random(50),r.random(50),c=r.random(50),s=r.uniform(10,100,50),cmap='plasma')
    return F

def canon(o,arr):
    if isinstance(o,dict):
        if "__arr__" in o:
            a=np.asarray(arr.d[o["__arr__"]])
            return "arr:"+hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:10]+str(a.shape)
        return {k:canon(v,arr) for k,v in o.items()}
    if isinstance(o,list): return [canon(v,arr) for v in o]
    return o
def norm(t,arr):
    t=json.loads(json.dumps(t))
    for k in ('report','replay_report','mpl','np'): t.pop(k,None)
    return canon(t,arr)
def px(f):
    b=io.BytesIO(); f.savefig(b,format='png',dpi=50); b.seek(0); return mi.imread(b)[...,:3]

print(f"{'recipe':16s} {'fix':5s} {'pixdiff':>8s}  issue")
bad=[]
for name,f in build().items():
    try:
        t1,a1=S.dump_figure(f); f2=S.replay_figure(t1,a1); t2,a2=S.dump_figure(f2)
        same=norm(t1,a1)==norm(t2,a2)
        p1,p2=px(f),px(f2)
        d=float(np.abs(p1-p2).mean()) if p1.shape==p2.shape else float('nan')
        rep=t1.get('report',[])+t1.get('replay_report',[])
        key=''
        for r in rep:
            if 'unknown' in r or 'failed' in r: key=r[:46]; break
        if d>0.01 or np.isnan(d): bad.append((name,d,key))
        print(f"{name:16s} {'ok' if same else 'DIFF':5s} {d:8.4f}  {key}")
    except Exception as e:
        bad.append((name,999,str(e)[:40])); print(f"{name:16s} {'ERR':5s} {'-':>8s}  {type(e).__name__}: {str(e)[:40]}")
print("\nPIXEL FAILURES (>0.01):",[(b[0],round(b[1],3)) for b in bad])
