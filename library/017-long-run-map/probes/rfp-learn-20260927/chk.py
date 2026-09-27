import numpy as np, pandas as pd, re
cols = ['fen','depth','sdump','alpha','beta','impr','ply','gid','static','q','vfin','dfin','nodes','best','bcap','mate'] + [f'v{i}' for i in range(1,13)]
t = pd.read_csv('lab.tsv', sep='\t', header=None, names=cols, na_values=['nan'])
X = np.fromfile('lab.f32', dtype='<f4').reshape(len(t), 163)
c = open('fitexport.out').read()
arr = lambda n: np.array([float(x) for x in re.search(n + r': \[f32; \d+\] = \[(.*?)\]', c).group(1).split(',')])
sc = lambda n: float(re.search(n + r': f32 = (\S+);', c).group(1))
HB, HW, ZB, ZW = sc('RFP_H_B'), arr('RFP_H_W'), sc('RFP_Z_B'), arr('RFP_Z_W')
sel = np.where((t.gid % 5 == 0) & (t.static >= t.beta) & (t.depth <= 7) & (t.mate == 0))[0][::997][:300]
with open('chk.tsv', 'w') as o:
    for i in sel:
        r = t.iloc[i]; f0 = r.fen.split()[0]
        npm = sum({'n':3,'b':3,'r':5,'q':9}.get(ch.lower(),0) for ch in f0); pw = sum(ch in 'pP' for ch in f0)
        g, d = float(r.static - r.beta), float(r.depth)
        f = np.array([np.log1p(g), g/d, d, np.log1p(g)/d, np.log1p(npm), pw, np.log1p(abs(r.static)), r.impr, np.log1p(r.ply)])
        h = HB + f @ HW; z = ZB + f @ ZW[:9] + X[i,96:160] @ ZW[9:]
        o.write(f'{r.fen}\t{int(r.depth)}\t{int(g)}\t{int(r.impr)}\t{int(r.ply)}\t{h:.4f}\t{z:.4f}\n')
print(len(sel))
