import subprocess, sys, os
from multiprocessing import Pool
ENG=sys.argv[1]; NET=sys.argv[2]; FENS=sys.argv[3]; OUT=sys.argv[4]; BUDGETS=[2000,64000]
def score(p, fen, n):
    p.stdin.write(f"ucinewgame\nposition fen {fen}\ngo nodes {n}\n"); p.stdin.flush()
    last=None
    while True:
        l=p.stdout.readline()
        if not l: return None
        if l.startswith("info") and " score " in l:
            t=l.split(); i=t.index("score")
            last=(t[i+1],int(t[i+2]))
        if l.startswith("bestmove"): return last
def work(chunk):
    p=subprocess.Popen([ENG,"--wdl",NET],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
    p.stdin.write("uci\nisready\n"); p.stdin.flush()
    while "readyok" not in p.stdout.readline(): pass
    out=[]
    for i,fen in chunk:
        r=[score(p,fen,n) for n in BUDGETS]
        out.append((i,r))
    p.stdin.write("quit\n"); p.stdin.flush(); return out
if __name__=="__main__":
    fens=[l.strip() for l in open(FENS)][:6000]
    chunks=[list(enumerate(fens))[k::6] for k in range(6)]
    with Pool(6) as pool: res=pool.map(work,chunks)
    with open(OUT,"w") as f:
        for part in res:
            for i,r in part:
                f.write(f"{i}\t"+"\t".join(f"{a}:{b}" if a else "none:0" for a,b in [x if x else (None,0) for x in r])+"\n")
