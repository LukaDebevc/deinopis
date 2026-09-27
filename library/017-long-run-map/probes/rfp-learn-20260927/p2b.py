# P2b, EXPLORATORY (written after p1.out was read). Two repairs to p1:
#  1. p1's Laplace fits started the log-scale bias at 0 (sigma = 1 cp) and never
#     converged (NLL gain < 0 is impossible for a model nesting the constant).
#     Here: closed-form OLS, and Laplace by full-batch LBFGS from the constant fit.
#  2. RFP is one-sided: it fails only when v < beta. q >= static by stand pat, so
#     a symmetric |error| scale cannot tell upside from downside. Fit the
#     downside directly: logistic P(v_own < beta) on candidate rows (static >= beta),
#     and compare with today's margin at the same prune count.
import sys, numpy as np, pandas as pd, torch
torch.manual_seed(0)
dev = 'cuda'
cols = ['fen','depth','sdump','alpha','beta','impr','ply','gid','static','q','vfin','dfin','nodes','best','bcap','mate'] + [f'v{i}' for i in range(1,13)]
t = pd.read_csv('lab.tsv', sep='\t', header=None, names=cols, na_values=['nan'])
X = np.fromfile('lab.f32', dtype='<f4').reshape(len(t), 163)
t['vown'] = t[[f'v{i}' for i in range(1,13)]].to_numpy()[np.arange(len(t)), np.clip(t.depth.to_numpy(),1,12)-1]
ok = (t.mate==0) & t.vown.notna() & (t.static.abs()<3000) & (t.vfin.abs()<3000) & (t.q.abs()<3000) & (t.vown.abs()<3000) & (t.depth<=7)
t = t[ok].reset_index(drop=True); X = X[ok.to_numpy()]
m = np.array([(sum({'n':3,'b':3,'r':5,'q':9}.get(c.lower(),0) for c in f.split()[0]), sum(c in 'pP' for c in f.split()[0])) for f in t.fen])
lg = torch.tensor(X[:,160:163]).softmax(1).numpy()
st, be, d, vo, q = (t[c].to_numpy().astype(float) for c in ['static','beta','depth','vown','q'])
dv = vo - st
val = (t.gid % 5 == 0).to_numpy(); es = (t.gid % 5 == 1).to_numpy(); tr = ~val & ~es
print(f'rows {len(t)} (depth<=7)  q>=static always: {np.all(q>=st)}')
print('\n== sign of the error, v_own - static')
for name, s in [('all', np.ones(len(t),bool)), ('q==static', q==st), ('q>static', q>st), ('static>=beta', st>=be)]:
    x = dv[s]; print(f'  {name:13s} n {s.sum():7d}  mean {x.mean():+7.1f}  median {np.median(x):+6.1f}  P(>0) {np.mean(x>0):.3f}  P(<-100) {np.mean(x<-100):.3f}  P(>+100) {np.mean(x>100):.3f}')

hand = np.c_[np.log1p(m[:,0]), m[:,1], d, np.log1p(np.abs(st)), t.impr, lg[:,1], np.log1p(t.ply)]
qf = np.c_[np.log1p(q-st), q!=st]
def Z(A):
    mu, sd = A[tr].mean(0), A[tr].std(0)+1e-6
    return torch.tensor(np.c_[np.ones(len(A)), (A-mu)/sd], dtype=torch.float64, device=dev)
print('\n== sigma refit (exact): held-out r2 on ln(|dev|+1) and Laplace NLL gain (nats/row)')
y = np.log(np.abs(dv)+1); ad = torch.tensor(np.abs(dv), device=dev)
c0 = np.abs(dv[tr]).mean(); lap0 = np.mean(np.abs(dv[val])/c0 + np.log(c0))
for k, A in [('hand',hand), ('hand+q',np.c_[hand,qf]), ('hand+zv',np.c_[hand,X[:,96:160]]), ('hand+all163',np.c_[hand,X]), ('hand+q+all163',np.c_[hand,qf,X])]:
    Zt = Z(A); w = torch.linalg.lstsq(Zt[tr], torch.tensor(y[tr],device=dev).unsqueeze(1)).solution
    p = (Zt@w).squeeze(1).cpu().numpy(); r2 = 1-np.mean((y[val]-p[val])**2)/np.var(y[val])
    wl = torch.zeros(Zt.shape[1],1,dtype=torch.float64,device=dev); wl.data[0]=np.log(c0); wl.requires_grad_(True)
    opt = torch.optim.LBFGS([wl], max_iter=500, line_search_fn='strong_wolfe'); itr = torch.tensor(np.where(tr)[0],device=dev)
    def cl():
        opt.zero_grad(); s=(Zt[itr]@wl).squeeze(1); l=(ad[itr]*torch.exp(-s)+s).mean(); l.backward(); return l
    opt.step(cl)
    with torch.no_grad():
        iv=torch.tensor(np.where(val)[0],device=dev); s=(Zt[iv]@wl).squeeze(1); lv=(ad[iv]*torch.exp(-s)+s).mean().item()
    print(f'  {k:14s} r2 {r2:.4f}   NLL gain {lap0-lv:+.4f}')

print('\n== downside classifier P(v_own < beta), candidate rows static >= beta, held-out')
cand = st >= be; fl = (vo < be).astype(float)
gap = st - be
gp = np.maximum(gap, 0); base_feats = np.c_[np.log1p(gp), gp/d, d, np.log1p(gp)/d]
vc = val & cand; nvc = vc.sum()
today = (gap >= 75*d); nb = (today & vc).sum(); wb = (today & vc & (vo<be)).sum()
print(f'  candidates in val {nvc}  fail-low rate {fl[vc].mean():.4f}   today: prunes {nb}  wrong {wb} ({wb/nb:.4%})')
F = torch.tensor(fl, device=dev, dtype=torch.float64)
def logit_fit(A, l2=1e-4):
    Zt = Z(A); itr = torch.tensor(np.where(tr&cand)[0],device=dev)
    w = torch.zeros(Zt.shape[1],1,dtype=torch.float64,device=dev,requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=1000, line_search_fn='strong_wolfe')
    def cl():
        opt.zero_grad(); l=torch.nn.functional.binary_cross_entropy_with_logits((Zt[itr]@w).squeeze(1),F[itr])+l2*(w[1:]**2).sum(); l.backward(); return l
    opt.step(cl)
    return (Zt@w).squeeze(1).detach().cpu().numpy()
arms = {'margin only': base_feats, 'margin+hand': np.c_[base_feats,hand], 'margin+hand+zv': np.c_[base_feats,hand,X[:,96:160]],
        'margin+hand+all163': np.c_[base_feats,hand,X], 'margin+hand+q': np.c_[base_feats,hand,qf], 'margin+hand+q+all163': np.c_[base_feats,hand,qf,X]}
for k, A in arms.items():
    p = logit_fit(A)[vc]; o = np.argsort(p); fv = (vo<be)[vc]
    w_same = fv[o[:nb]].sum(); more = int((np.cumsum(fv[o]) <= wb).sum())
    auc_rank = np.empty(len(p)); auc_rank[o]=np.arange(len(p)); n1=fv.sum(); n0=len(fv)-n1
    auc = (auc_rank[fv].sum()-n1*(n1-1)/2)/(n1*n0)
    print(f'  {k:22s} AUC {auc:.4f}   same prunes -> wrong {w_same:5d} ({(w_same-wb)/wb:+.1%})   same wrong -> prunes {more:6d} ({(more-nb)/nb:+.1%})')
print('DONE')
