import numpy as np
S="/tmp/claude-1000/-home-luka-Desktop-chess/cdfda02d-5fbc-4787-8d4a-1edce05b1493/scratchpad/"
rows=[l.rstrip('\n').split('|') for l in open(S+"wdl20k.out")]
fens=[l.strip() for l in open(S+"wdl20k.fens")]
cp=np.array([int(r[0]) for r in rows],float)
lg=np.array([[float(x) for x in r[-1].split(',')] for r in rows]); p=np.exp(lg-lg.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
L,D,W=p.T; E=W+D/2
X768=np.zeros((len(rows),768),np.float32)
for i,r in enumerate(rows):
    for f in r[1].split(','): X768[i,int(f)]=1
cnt=np.stack([X768[:,s*384+64*t:s*384+64*t+64].sum(1) for s in (0,1) for t in range(5)],1)
vals=np.array([1,3,3,5,9]*2); npm=(cnt[:,[1,2,3,4,6,7,8,9]]*vals[[1,2,3,4,6,7,8,9]]).sum(1)
from sklearn.ensemble import HistGradientBoostingRegressor as H
m=H(max_iter=400,learning_rate=0.05).fit(np.c_[E,cnt][6000:],D[6000:])
Dres=D-m.predict(np.c_[E,cnt])
# searches
V=np.full((20000,2),np.nan); ok=np.zeros(20000,bool)
for l in open(S+"sigma6k.tsv"):
    t=l.split('\t'); i=int(t[0]); a=[x.split(':') for x in t[1:]]
    if all(k=='cp' for k,_ in a): V[i]=[int(v) for _,v in a]; ok[i]=True
idx=np.where(ok)[0]; print("usable (no mate either budget):",len(idx),"of 6000")
big=np.abs(V[idx]).max(1)<20000; idx=idx[big]; print("dropped mate-range scores, kept",len(idx))
dev=np.clip(V[idx,0]-V[idx,1],-1500,1500); dst=np.clip(cp[idx]-V[idx,1],-1500,1500)
print("\nraw rows: static  V2k  V64k  D  Dres  npm  fen")
for j in idx[:10]: print("%6d %6d %6d  %.2f %+.3f %3d  %s"%(cp[j],V[j,0],V[j,1],D[j],Dres[j],npm[j],fens[j][:50]))
print("\nsd(V2k-V64k) overall %.1f cp, mean|dev| %.1f ; sd(static-V64k) %.1f"%(dev.std(),np.abs(dev).mean(),dst.std()))
def quint(x,y,name):
    q=np.quantile(x,[.2,.4,.6,.8]); b=np.digitize(x,q)
    sds=[y[b==k].std() for k in range(5)]
    print("%-22s "%name+"  ".join("%6.1f"%s for s in sds)+"   Q5/Q1 %.2f"%(sds[4]/sds[0]))
print("\nsd of (V2k - V64k) by quintile (Q1 low .. Q5 high)")
feats={"non-pawn material":npm[idx],"D (draw prob)":D[idx],"D residual | E,counts":Dres[idx],"|E-0.5|":np.abs(E[idx]-.5),"entropy WDL":-(p[idx]*np.log(p[idx]+1e-9)).sum(1),"|static - V2k|":np.abs(cp[idx]-V[idx,0])}
for k,x in feats.items(): quint(x,dev,k)
print("\nwithin non-pawn-material quartiles: sd ratio (top half / bottom half of feature)")
qm=np.quantile(npm[idx],[.25,.5,.75]); bm=np.digitize(npm[idx],qm)
for k in ["D (draw prob)","D residual | E,counts","|E-0.5|","|static - V2k|"]:
    x=feats[k]; rs=[]
    for g in range(4):
        s=bm==g; med=np.median(x[s]); hi=dev[s][x[s]>med].std(); lo=dev[s][x[s]<=med].std(); rs.append(hi/lo)
    print("%-22s "%k+"  ".join("%.2f"%r for r in rs))
# regression on ln|dev|
from sklearn.linear_model import LinearRegression as LR
y=np.log(np.abs(dev)+5)
def fit(cols,name):
    Xf=np.column_stack(cols); h=len(y)//2; mdl=LR().fit(Xf[:h],y[:h]); pr=mdl.predict(Xf[h:]); print("r2 ln|dev| %-40s %.4f"%(name,1-((y[h:]-pr)**2).mean()/y[h:].var()))
ln=np.log
fit([ln(npm[idx]+1)],"log npm")
fit([ln(npm[idx]+1),D[idx]],"log npm + D")
fit([ln(npm[idx]+1),Dres[idx]],"log npm + Dres")
fit([ln(npm[idx]+1),ln(np.abs(cp[idx]-V[idx,0])+5)],"log npm + log|static-V2k|")
fit([ln(npm[idx]+1),ln(np.abs(cp[idx]-V[idx,0])+5),D[idx],Dres[idx],np.abs(E[idx]-.5)],"all")
