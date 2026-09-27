# P1/P2 of the uncertainty-head probe, 2026-09-27. Reads P0's lab.tsv/lab.f32.
#
# PRE-REGISTERED (written before any labelled row was read):
#  P1 sigma: target y = ln(|v_own - static| + 1), v_own = the node's own-depth
#    iteration of a 64k-node search. Held-out = 6 of 32 dump files (disjoint
#    games). The net earns P2/P3 only if linear(net layers + hand) beats
#    linear(hand) by >= 0.02 r2 AND shows a Laplace-NLL gain. hand = free
#    in-engine features: log npm, pawns, depth, log1p|static|, improving, D, ply.
#  hand+q adds |q - static| (costs a qsearch): the reference the net should reach.
#  P2: RFP replay on held-out rows: today's rule prunes when
#    static - 75*d >= beta. The sigma rule prunes when (static - beta) >= k*sighat,
#    k swept to match the prune rate. Wrong prune = v_own < beta. Report wrong
#    prunes at matched rate and prunes at matched wrong-count. A proxy, not Elo.
#  Capture / q-gap heads: logistic, held-out AUC vs the same hand features.
import sys, numpy as np, pandas as pd, torch
torch.manual_seed(0); np.random.seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
D = sys.argv[1] if len(sys.argv) > 1 else '.'
cols = ['fen','depth','sdump','alpha','beta','impr','ply','gid','static','q','vfin','dfin','nodes','best','bcap','mate'] + [f'v{i}' for i in range(1,13)]
t = pd.read_csv(f'{D}/lab.tsv', sep='\t', header=None, names=cols, na_values=['nan'])
X = np.fromfile(f'{D}/lab.f32', dtype='<f4').reshape(len(t), 163)
print(f'rows {len(t)}  files {t.gid.nunique()}  static==dump {np.mean(t.static==t.sdump):.4f}')
vown = t[[f'v{i}' for i in range(1,13)]].to_numpy()[np.arange(len(t)), np.clip(t.depth.to_numpy(),1,12)-1]
t['vown'] = vown
ok = (t.mate==0) & t.vown.notna() & (t.static.abs()<3000) & (t.vfin.abs()<3000) & (t.q.abs()<3000) & (t.vown.abs()<3000)
print(f'kept {ok.mean():.4f} (mate / |score|>=3000 / missing own-depth iteration dropped)')
t = t[ok].reset_index(drop=True); X = X[ok.to_numpy()]
print('depth histogram', t.depth.value_counts().sort_index().to_dict())

def mat(fen):
    b = fen.split()[0]; v = {'n':3,'b':3,'r':5,'q':9}
    return sum(v.get(c.lower(),0) for c in b), sum(c in 'pP' for c in b)
m = np.array([mat(f) for f in t.fen]); npm, pawns = m[:,0], m[:,1]
lg = torch.tensor(X[:,160:163]).softmax(1).numpy()   # (L, D, W)
dv = (t.vown - t.static).to_numpy().astype(float)
y = np.log(np.abs(dv) + 1)
qg = (t.q - t.static).to_numpy().astype(float)
hand = np.c_[np.log1p(npm), pawns, t.depth, np.log1p(np.abs(t.static)), t.impr, lg[:,1], np.log1p(t.ply)]
handq = np.c_[hand, np.log1p(np.abs(qg)), qg != 0]
val = (t.gid % 5 == 0).to_numpy(); es = (t.gid % 5 == 1).to_numpy(); tr = ~val & ~es   # es = early-stopping files, never reported
print(f'train {tr.sum()}  early-stop {es.sum()}  val {val.sum()}   sd(v_own-static) {dv.std():.1f} cp   median |dev| {np.median(np.abs(dv)):.0f}   q!=static {np.mean(qg!=0):.3f}')
ab = np.abs(t.static.to_numpy())
print('median |dev| by |static| bin:', {f'<{b}': round(float(np.median(np.abs(dv[(ab>=a)&(ab<b)]))),0) for a,b in [(0,50),(50,150),(150,400),(400,1000),(1000,3000)]})
print(f'sd(v_own-static) by depth:', {int(d): round(float(dv[t.depth==d].std()),1) for d in range(1,8)})

sets = {
  'hand': hand, 'hand+q': handq,
  'hand+zv(l2)': np.c_[hand, X[:,96:160]], 'hand+hv(up)': np.c_[hand, X[:,32:96]],
  'hand+cu(mid)': np.c_[hand, X[:,0:32]], 'hand+all163': np.c_[hand, X],
  'hand+q+all163': np.c_[handq, X],
}
def std(A):
    mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-6
    return torch.tensor((A-mu)/sd, dtype=torch.float32, device=dev)
def fit(A, loss, out=1, hidden=0, steps=3000, lr=3e-3, wd=1e-4):
    Z = std(A); n = Z.shape[1]
    net = torch.nn.Linear(n, out) if hidden == 0 else torch.nn.Sequential(torch.nn.Linear(n,hidden), torch.nn.ReLU(), torch.nn.Linear(hidden,out))
    net = net.to(dev); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    itr = torch.tensor(np.where(tr)[0], device=dev); best=(1e9,None)
    for s in range(steps):
        idx = itr[torch.randint(len(itr), (65536,), device=dev)]
        opt.zero_grad(); l = loss(net(Z[idx]).squeeze(-1), idx); l.backward(); opt.step()
        if s % 250 == 249 or s == steps-1:
            with torch.no_grad():
                ei = torch.tensor(np.where(es)[0], device=dev); le = loss(net(Z[ei]).squeeze(-1), ei).item()
            if le < best[0]:
                vi = torch.tensor(np.where(val)[0], device=dev)
                best = (le, loss(net(Z[vi]).squeeze(-1), vi).item(), net(Z).squeeze(-1).detach().cpu().numpy())
    return best[1:]
Y = torch.tensor(y, dtype=torch.float32, device=dev)
ADV = torch.tensor(np.abs(dv), dtype=torch.float32, device=dev)
mse = lambda p, i: ((p - Y[i])**2).mean()
lap = lambda s, i: (ADV[i] * torch.exp(-s) + s).mean()     # Laplace NLL, s = log scale
r2 = lambda p: 1 - np.mean((y[val]-p[val])**2) / np.var(y[val])
lap0 = np.mean(np.abs(dv[val]) / np.abs(dv[tr]).mean() + np.log(np.abs(dv[tr]).mean()))
print('\n== P1 sigma: held-out r2 on ln(|v_own-static|+1), Laplace NLL gain vs constant scale (nats/row)')
sig = {}
for k, A in sets.items():
    _, p = fit(A, mse); lv, s = fit(A, lap); sig[k] = np.exp(s)
    print(f'  {k:16s} r2 {r2(p):.4f}   NLL gain {lap0 - lv:+.4f}')
for k in ['hand+all163', 'hand+q+all163']:
    _, p = fit(sets[k], mse, hidden=64, steps=6000); lv, s = fit(sets[k], lap, hidden=64, steps=6000)
    print(f'  {k+" MLP64":16s} r2 {r2(p):.4f}   NLL gain {lap0 - lv:+.4f}'); sig[k+' MLP64'] = np.exp(s)

def auc(p, lab):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(len(p)); n1 = lab.sum(); n0 = len(lab)-n1
    return (r[lab==1].sum() - n1*(n1-1)/2) / (n1*n0)
for name, lab in [('best move is a capture', t.bcap.to_numpy()), ('q != static', (qg!=0).astype(int))]:
    L = torch.tensor(lab, dtype=torch.float32, device=dev)
    bce = lambda p, i: torch.nn.functional.binary_cross_entropy_with_logits(p, L[i])
    print(f'\n== {name}: base rate {lab.mean():.3f}, held-out AUC')
    for k in ['hand', 'hand+q', 'hand+zv(l2)', 'hand+all163'] if name.startswith('best') else ['hand', 'hand+zv(l2)', 'hand+all163']:
        _, p = fit(sets[k], bce); print(f'  {k:16s} AUC {auc(p[val], lab[val]):.4f}')

print('\n== P2 RFP replay on held-out rows (wrong = v_own < beta)')
v = val & (t.depth <= 7).to_numpy()
st, be, d, vo = t.static.to_numpy()[v], t.beta.to_numpy()[v], t.depth.to_numpy()[v], t.vown.to_numpy()[v]
base = st - 75*d >= be; nb, wb = base.sum(), (base & (vo < be)).sum()
print(f'  rows {v.sum()}   today: prunes {nb} ({nb/v.sum():.3f})  wrong {wb} ({wb/max(nb,1):.4f} of prunes)')
for k in ['hand', 'hand+q', 'hand+zv(l2)', 'hand+all163', 'hand+all163 MLP64', 'hand+q+all163 MLP64']:
    z = (st - be) / sig[k][v]
    zs = np.sort(z)[::-1]
    thr = zs[nb-1]; pr = z >= thr; w_at = (pr & (vo < be)).sum()
    ok_ = np.cumsum(vo[np.argsort(-z)] < be[np.argsort(-z)]) <= wb
    more = int(ok_.sum())
    print(f'  {k:20s} same prunes -> wrong {w_at:5d} ({(w_at-wb)/max(wb,1):+.1%})   same wrong -> prunes {more:6d} ({(more-nb)/max(nb,1):+.1%})')
print('DONE')
