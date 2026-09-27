# Fit the deployable RFP classifiers and print them as Rust constants.
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
for name, A in [('H', f), ('Z', np.c_[f, X[:,96:160]])]:
    b, w, lo = fit(A)
    o = np.argsort(lo[val]); fv = fl[val]
    thr_same_prunes = np.sort(lo[val])[nb-1]
    more = int((np.cumsum(fv[o]) <= wb).sum()); thr_same_wrong = lo[val][o][more-1]
    print(f'\n// arm {name}: same prunes -> wrong {fv[o[:nb]].sum()} (today {wb}); same wrong -> prunes {more} (today {nb})')
    print(f'// thr at same prunes {thr_same_prunes:.4f}, at same wrong {thr_same_wrong:.4f}')
    print(f'const RFP_{name}_B: f32 = {b:.6e};')
    print(f'const RFP_{name}_W: [f32; {len(w)}] = [' + ', '.join(f'{x:.6e}' for x in w) + '];')
    # spot check on the first 5 val rows, for the engine-side verifier
    i5 = np.where(val)[0][:5]
    print(f'// check logits {[round(float(lo[i]),4) for i in i5]}  fens {[t.fen[i] for i in i5[:2]]}')
