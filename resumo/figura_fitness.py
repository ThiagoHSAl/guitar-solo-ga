import json,sys,numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":7.5,"axes.titlesize":8})
R=json.load(open(sys.argv[1])); out=sys.argv[2]
def arr(rs,key):
    L=max(len(r["history"][key]) for r in rs)
    return np.array([r["history"][key]+[r["history"][key][-1]]*(L-len(r["history"][key])) for r in rs])
fig,(a,b)=plt.subplots(2,1,figsize=(3.3,3.0),dpi=300)
base=[r for r in R if r["exp"]=="base"]
B,M,W=arr(base,"best"),arr(base,"mean"),arr(base,"worst")
x=np.arange(B.shape[1])
a.fill_between(x,W.mean(0),B.mean(0),color="#4C72B0",alpha=.12,lw=0,label="pior–melhor")
a.plot(x,B.mean(0),color="#4C72B0",lw=1.4,label="melhor")
a.fill_between(x,B.mean(0)-B.std(0),B.mean(0)+B.std(0),color="#4C72B0",alpha=.3,lw=0)
a.plot(x,M.mean(0),color="#4C72B0",lw=1,ls="--",label="média da população")
a.set_title("(a) Blues shuffle em Lá, 5 sementes")
a.set_ylabel("Fitness"); a.set_ylim(0.5,1); a.grid(alpha=.3); a.legend(loc="lower right",fontsize=6.5)
groups=[("base","mutação 0,3 (padrão)","#4C72B0","-"),("mut0.1","mutação 0,1","#55A868","-"),
        ("mut0.6","mutação 0,6","#C44E52","-"),("uniform","inicialização uniforme","#8172B2","--"),
        ("nolick","sem licks","#937860",":")]
for e,l,c,ls in groups:
    rs=[r for r in R if r["exp"]==e and r["seed"]<=3]
    b.plot(arr(rs,"best").mean(0),color=c,ls=ls,lw=1.2,label=l)
b.set_title("(b) Melhor indivíduo por configuração, 3 sementes")
b.set_xlabel("Geração"); b.set_ylabel("Fitness"); b.set_ylim(0.4,0.99); b.grid(alpha=.3); b.legend(loc="lower right",fontsize=6,ncol=2,framealpha=0.9)
from matplotlib.ticker import FuncFormatter
for ax in (a, b):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
fig.tight_layout(); fig.savefig(out); print(out)
