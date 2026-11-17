import io,sys,math,time
sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
import torch,torch.nn as nn,torch.nn.functional as F
torch.backends.cudnn.benchmark=True
try: torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
except: pass
dev="cuda"
print("GPU:",torch.cuda.get_device_name(0),"torch",torch.__version__,flush=True)
F_IN=256; d=96; nblk=3; nh=8; ffm=2; B=2048; drop=0.1
class Blk(nn.Module):
  def __init__(s):
    super().__init__(); s.n1=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d); s.o=nn.Linear(d,d)
    s.n2=nn.LayerNorm(d); s.ff=nn.Sequential(nn.Linear(d,ffm*d),nn.GELU(),nn.Linear(ffm*d,d)); s.nh=nh; s.p=drop
  def forward(s,x):
    h=s.n1(x); Bb,T,D=h.shape; q,k,v=s.qkv(h).chunk(3,-1)
    q=q.view(Bb,T,s.nh,-1).transpose(1,2); k=k.view(Bb,T,s.nh,-1).transpose(1,2); v=v.view(Bb,T,s.nh,-1).transpose(1,2)
    a=F.scaled_dot_product_attention(q,k,v,dropout_p=s.p if s.training else 0.0).transpose(1,2).reshape(Bb,T,D)
    x=x+s.o(a); x=x+s.ff(s.n2(x)); return x
class FT(nn.Module):
  def __init__(s):
    super().__init__(); s.W=nn.Parameter(torch.randn(F_IN,d)*0.02); s.b=nn.Parameter(torch.zeros(F_IN,d))
    s.cls=nn.Parameter(torch.randn(1,1,d)*0.02); s.blks=nn.ModuleList([Blk() for _ in range(nblk)])
    s.ln=nn.LayerNorm(d); s.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,1))
  def forward(s,x):
    Bb=x.shape[0]; tok=x[:,:,None]*s.W[None]+s.b[None]
    h=torch.cat([s.cls.expand(Bb,-1,-1),tok],1)
    for blk in s.blks: h=blk(h)
    return s.head(s.ln(h[:,0]))[:,0]
m=FT().to(dev); opt=torch.optim.AdamW(m.parameters(),1e-3)
x=torch.randn(B,F_IN,device=dev); yb=torch.randn(B,device=dev)
print("warmup 15 step ...",flush=True)
for _ in range(15):
  opt.zero_grad(); l=F.huber_loss(m(x),yb); l.backward(); opt.step()
torch.cuda.synchronize()
t=time.time()
for _ in range(40):
  opt.zero_grad(); l=F.huber_loss(m(x),yb); l.backward(); opt.step()
torch.cuda.synchronize(); tr=(time.time()-t)/40
m.eval(); t=time.time()
with torch.no_grad():
  for _ in range(40): _=m(x)
torch.cuda.synchronize(); fw=(time.time()-t)/40
N=3783989; sfit=math.ceil(0.8*N/B); sval=math.ceil(0.2*N/B); ep=sfit*tr+sval*fw
print(f"★train {tr*1000:.0f}ms/step  fwd {fw*1000:.0f}ms/step",flush=True)
print(f"1エポック={ep:.0f}s ({ep/60:.1f}分)  15ep想定x15fold={ep*15*15/3600:.1f}時間",flush=True)
