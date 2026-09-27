# p2c, EXPLORATORY, after the R3 screen lost (H -6.8, Z -24.3): where do the learned
# rules prune, by depth? The replay counted prunes as equal; a depth-d prune saves ~1.88^d.
# Refits exactly as fitexport.py.
# Features exactly as src/search.rs computes them at the RFP block:
#   f = [ln(1+g), g/d, d, ln(1+g)/d, ln(1+npm), pawns, ln(1+|static|), improving, ln(1+ply)]
#   g = static - beta (>= 0 on candidate rows), d = depth, npm = non-pawn material
#   in pawns (N=B=3, R=5, Q=9, both sides), pawns = count, both sides.
# Arm H: f only. Arm Z: f + zv (the net's l2 output, 64 numbers).
# logit(P(v_own < beta)) = b + w.f (+ u.zv); prune when logit < thr.
import numpy as np, pandas as pd, torch
dev = 'cuda'
cols = ['fen','depth','sdump','alpha','beta','impr','ply','gid','static','q','vfin','dfin','nodes','best','bcap','mate'] + [f'v{i}' for i in range(1,13)]
t = pd.read_csv('lab.tsv', sep='\t', header=None, names=cols, na_values=['nan'])
X = np.fromfile('lab.f32', dtype='<f4').reshape(len(t), 163)
t['vown'] = t[[f'v{i}' for i in range(1,13)]].to_numpy()[np.arange(len(t)), np.clip(t.depth.to_numpy(),1,12)-1]
ok = (t.mate==0) & t.vown.notna() & (t.static.abs()<3000) & (t.vown.abs()<3000) & (t.depth<=7) & (t.static>=t.beta)
t = t[ok].reset_index(drop=True); X = X[ok.to_numpy()]
m = np.array([(sum({'n':3,'b':3,'r':5,'q':9}.get(c.lower(),0) for c in f.split()[0]), sum(c in 'pP' for c in f.split()[0])) for f in t.fen])
st, be, d = (t[c].to_numpy().astype(float) for c in ['static','beta','depth'])
g = st - be
f = np.c_[np.log1p(g), g/d, d, np.log1p(g)/d, np.log1p(m[:,0]), m[:,1], np.log1p(np.abs(st)), t.impr, np.log1p(t.ply)]
fl = (t.vown < t.beta).to_numpy()
val = (t.gid % 5 == 0).to_numpy(); tr = ~val        # no early stopping needed: convex, LBFGS, small l2
F = torch.tensor(fl, dtype=torch.float64, device=dev)
def fit(A, l2=1e-4):
    mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-9
    Zt = torch.tensor(np.c_[np.ones(len(A)), (A-mu)/sd], dtype=torch.float64, device=dev)
    it = torch.tensor(np.where(tr)[0], device=dev)
    w = torch.zeros(Zt.shape[1], 1, dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=2000, line_search_fn='strong_wolfe')
    def cl():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits((Zt[it]@w).squeeze(1), F[it]) + l2*(w[1:]**2).sum(); l.backward(); return l
    opt.step(cl)
    w = w.detach().cpu().numpy().ravel()
    raw = w[1:] / sd; b = w[0] - (w[1:] * mu / sd).sum()
    return b, raw, A @ raw + b
today = g >= 75*d; vt = val & today
nb, wb = vt.sum(), (vt & fl).sum()
print(f'val candidates {val.sum()}  today prunes {nb} wrong {wb}')
ebf = 1.88
fv = fl[val]; dv_ = d[val]
print(f'{"rule":8s}' + ''.join(f'  d{k}:prune/wrong' for k in range(1,8)) + '   saved(sum 1.88^d)  wrong-weighted')
def row(name, pr):
    cells = ''.join(f'  {int((pr&(dv_==k)).sum()):6d}/{int((pr&(dv_==k)&fv).sum()):4d}' for k in range(1,8))
    print(f'{name:8s}{cells}   {np.sum(ebf**dv_[pr]):12.0f}   {np.sum(ebf**dv_[pr&fv]):10.0f}')
row('today', today[val])
for name, A in [('H', f), ('Z', np.c_[f, X[:,96:160]])]:
    b, w, lo = fit(A); lv = lo[val]
    thr = np.sort(lv)[nb-1]; row(name+' same#', lv <= thr)
    pd_ = np.zeros(len(lv), bool)
    for k in range(1,8):
        idx = np.where(dv_==k)[0]; n_k = int((today[val] & (dv_==k)).sum())
        if n_k: pd_[idx[np.argsort(lv[idx])[:n_k]]] = True
    row(name+' perd', pd_)
# Per-depth thresholds on ALL candidate rows: the logit quantile that prunes today's
# fraction at each depth. For the engine's rfp_learn = 3 (H) / 4 (Z).
for name, A in [('H', f), ('Z', np.c_[f, X[:,96:160]])]:
    b, w, lo = fit(A); th = []
    for k in range(1,8):
        idx = d==k; frac = today[idx].mean(); th.append(float(np.quantile(lo[idx], frac)))
    print(f'const RFP_{name}_THR: [f32; 7] = [' + ', '.join(f'{x:.4f}' for x in th) + '];  // depth 1..7')
print('DONE')
