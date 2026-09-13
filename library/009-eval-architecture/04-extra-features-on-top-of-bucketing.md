## 4. Extra features *on top of* bucketing (`runs/extras-buck.log`)

Bar = `material+count` on the read, 0.017690 (replicates `psqtbuckets`'
0.017701 to 0.06%).

| arm | val | vs bar | same feature unbucketed |
|---|---|---|---|
| bar [material+count] | 0.017690 | — | — |
| **+pawnfile** | 0.016835 | −4.83% | −4.68% |
| +pawnpair | 0.016907 | −4.43% | −5.14% |
| **+pawnfile+pawnpair** | **0.016545** | **−6.47%** | −7.05% |
| +tile23 | 0.017143 | −3.09% | −4.85% |
| +centre | 0.017546 | −0.81% | −1.45% |

**Pawn features are complementary to bucketing; tiles and centre are partly
redundant with it.** Pawn structure keeps ~92% of its unbucketed relative gain
once the model is conditioned on material+count; tile23 keeps 64%, centre 56%.

**Best model of the session: 0.016545** = quadratic r=512 + material+count read
buckets + pawnfile + pawnpair, i.e. **−21.6% against the bar**.
