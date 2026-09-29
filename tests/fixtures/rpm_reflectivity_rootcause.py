"""Root cause of the toy reflectivity range / DC offset (docs/rock-physics-port.md).

Rebuilds the Rust raw 15 deg reflectivity (``filters_demo`` output
``angle_rfc.f32`` + ``labels.u8``) in numpy from the labels, the
``rpm_example`` trends and the Zoeppritz formula of
``datagenerator/zoeppritz_kernel.py``, checks it is bit-identical to the Rust
output, then swaps the inputs: 4 m per sample (legacy ``digi``), and the
legacy depth model (per-layer depth below mudline, water properties).

    cargo run --release -p synthoseis-core --example filters_demo -- /tmp/fd 7 4 64 64 128 4 30 3
    python tests/fixtures/rpm_reflectivity_rootcause.py /tmp/fd
"""
import json
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ricker_demo"
ni,nj,nk=json.load(open(f'{D}/meta.json'))['shape']
lab=np.fromfile(f'{D}/labels.u8','u1').reshape(ni,nj,nk)
rust=np.fromfile(f'{D}/angle_rfc.f32','<f4').reshape(ni,nj,nk)

def T(z):
    return dict(
      shale=(-0.00013*z**2+1.13*z+1580, -0.0001*z**2+0.96*z+279, 7.7e-12*z**3-8.8e-08*z**2+0.0004*z+1.957),
      brine=(-1.34e-05*z**2+0.49*z+2317, -1.0785e-05*z**2+0.391*z+1007, -7.8e-09*z**2+0.00012*z+2.021),
      oil=(-8.876e-06*z**2+0.505*z+1998, -1.126e-05*z**2+0.391*z+1036, -9.23e-09*z**2+0.00014*z+1.916))

def zoep(vp1,vs1,r1,vp2,vs2,r2,ang):
    th=np.deg2rad(ang)+0j; p=np.sin(th)/vp1
    t2=np.arcsin(p*vp2+0j); f1=np.arcsin(p*vs1+0j); f2=np.arcsin(p*vs2+0j)
    s1=np.sin(f1)**2; s2=np.sin(f2)**2; ct=np.cos(th); ct2=np.cos(t2); c1=np.cos(f1); c2=np.cos(f2)
    a=r2*(1-2*s2)-r1*(1-2*s1); b=r2*(1-2*s2)+2*r1*s1; c=r1*(1-2*s1)+2*r2*s2; d=2*(r2*vs2**2-r1*vs1**2)
    e=b*ct/vp1+c*ct2/vp2; f=b*c1/vs1+c*c2/vs2; g=a-d*ct/vp1*c2/vs2; h=a-d*ct2/vp2*c1/vs1
    det=e*f+g*h*p*p
    return ((f*(b*ct/vp1-c*ct2/vp2)-h*p*p*(a+det*ct/vp1*c2/vs2))/det).real.astype(np.float32)

def props(depth, water=False):
    # depth: (ni,nj,nk) metres
    t=T(depth); vp=np.empty(depth.shape); vs=np.empty(depth.shape); rho=np.empty(depth.shape)
    for code,name in ((0,'shale'),(1,'brine')):
        m=lab==code; vp[m],vs[m],rho[m]=(x[m] for x in t[name])
    m=lab>1; vp[m],vs[m],rho[m]=(x[m] for x in t['oil'])
    if water: vp[lab==255],vs[lab==255],rho[lab==255]=1500.,1000.,1.028
    return vp.astype(np.float32).astype(float),vs.astype(np.float32).astype(float),rho.astype(np.float32).astype(float)

def rfc(vp,vs,rho,ang=15.0):
    out=np.zeros(vp.shape,np.float32)
    out[...,:-1]=zoep(vp[...,:-1],vs[...,:-1],rho[...,:-1],vp[...,1:],vs[...,1:],rho[...,1:],ang)
    return out

k=np.arange(nk,dtype=float)
cases={}
cases['rust_current (100 m/sample, per-sample depth, water->oil)']=props(np.broadcast_to(k*100.0,lab.shape).copy())
cases['4 m/sample (legacy digi), per-sample depth']=props(np.broadcast_to(k*4.0,lab.shape).copy())
# legacy depth model: constant per label run, (run base - seabed) * digi (TVDML)
seabed=np.argmax(lab!=255,axis=-1)  # first non-water sample
dl=np.zeros(lab.shape)
for i in range(ni):
  for j in range(nj):
    L=lab[i,j]; start=0
    for kk in range(1,nk+1):
      if kk==nk or L[kk]!=L[start]:
        dl[i,j,start:kk]=max(0,kk-1-seabed[i,j])*4.0   # base of run, below seabed, digi 4 m
        start=kk
cases['legacy-style: per-layer TVDML x 4 m + water props']=props(dl,water=True)
res={}
for name,(vp,vs,rho) in cases.items():
    r=rfc(vp,vs,rho)
    if name.startswith('rust_current'):
        print('reproduces rust angle_rfc bit-exact?', np.array_equal(r.view(np.uint32), rust.view(np.uint32)), 'maxdiff', np.abs(r-rust).max())
    rr=r[...,:-1]
    res[name]=dict(min=float(rr.min()),max=float(rr.max()),mean=float(rr.mean()),median=float(np.median(rr)),
                   frac_abs_gt1=float((np.abs(rr)>1).mean()),frac_nonzero=float((rr!=0).mean()),
                   min_vp=float(vp.min()),vs_over_vp_max=float((vs/vp).max()))
    np.save(f'{D}/rootcause_rfc_{len(res)}.npy', r)
print(json.dumps(res,indent=1))
i,j,kk=np.unravel_index(np.argmin(rust),rust.shape)
vp,vs,rho=cases['rust_current (100 m/sample, per-sample depth, water->oil)']
print('min sample', (i,j,kk), 'labels', lab[i,j,kk], lab[i,j,kk+1], 'upper vp/vs/rho', vp[i,j,kk],vs[i,j,kk],rho[i,j,kk], 'lower', vp[i,j,kk+1],vs[i,j,kk+1],rho[i,j,kk+1], 'p*vp2', np.sin(np.deg2rad(15))/vp[i,j,kk]*vp[i,j,kk+1])
# where shale trend breaks
z=np.arange(0,13000,10.0); t=T(z)
print('shale vp vertex z', 1.13/(2*0.00013), 'shale vp=0 at', z[np.argmax(t['shale'][0]<=0)])
