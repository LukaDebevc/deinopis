import numpy as np, sys
rows=[l.rstrip('\n').split('|') for l in open(sys.argv[1])]
cp=np.array([int(r[0]) for r in rows],float)
lg=np.array([[float(x) for x in r[-1].split(',')] for r in rows])
p=np.exp(lg-lg.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
L,D,W=p[:,0],p[:,1],p[:,2]
E=W+D/2
X768=np.zeros((len(rows),768),np.float32)
for i,r in enumerate(rows):
    for f in r[1].split(','): X768[i,int(f)]=1
cnt=np.stack([X768[:,s*384+64*t:s*384+64*t+64].sum(1) for s in (0,1) for t in range(5)],1)  # P N B R Q per side
vals=np.array([1,3,3,5,9]*2); npm=(cnt[:,[1,2,3,4,6,7,8,9]]*vals[[1,2,3,4,6,7,8,9]]).sum(1)
pawns=cnt[:,0]+cnt[:,5]
print("n",len(rows),"mean D %.3f sd D %.3f"%(D.mean(),D.std()), "sanity cp vs 288.5*logit(E): corr %.4f"%np.corrcoef(cp,288.5*np.log(np.clip(E,1e-6,1-1e-6)/np.clip(1-E,1e-6,1)))[0,1])
rng=np.random.default_rng(0); idx=rng.permutation(len(rows)); tr,te=idx[:len(idx)//2],idx[len(idx)//2:]
from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.linear_model import Ridge
def r2(Xf,y,model):
    model.fit(Xf[tr],y[tr]); pr=model.predict(Xf[te]); return 1-((y[te]-pr)**2).mean()/y[te].var(), pr
a=np.abs(E-0.5)
sets={
 "E only":np.c_[E],
 "E + npm + pawns":np.c_[E,npm,pawns],
 "E + 10 piece counts":np.c_[E,cnt],
 "E + counts + 768 occupancy (GBM)":np.c_[E,cnt,X768],
}
res={}
for k,Xf in sets.items():
    s,pr=r2(Xf,D,H(max_iter=400,learning_rate=0.05))
    res[k]=pr; print("R2(D | %-34s) = %.4f   resid sd %.4f"%(k,s,np.sqrt(((D[te]-pr)**2).mean())))
# how big is the position-specific part in cp terms? readout alternatives
A=np.log(W/L)
print("corr(logit E, A=log W/L) %.4f"%np.corrcoef(np.log(E/(1-E)),A)[0,1])
# where readouts disagree: drawish positions
for lo,hi in [(0,.3),(.3,.6),(.6,.8),(.8,1.01)]:
    m=(D>=lo)&(D<hi)
    if m.sum()<50: continue
    lgE=np.log(E[m]/(1-E[m])); Am=A[m]/2
    print("D in [%.1f,%.1f): n=%5d  sd(288.5*logitE)=%.0f  sd(288.5*A/2)=%.0f  ratio A/2 : logitE slope %.2f"%(lo,hi,m.sum(),288.5*lgE.std(),288.5*Am.std(),np.polyfit(lgE,Am,1)[0]))
