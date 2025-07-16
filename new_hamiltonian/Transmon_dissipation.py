import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time as t
import sys
from tqdm import tqdm

at=t.time()

sites=4
nmax=4
# tsteps=10000000
tsteps=10000
dt=.0002

#real parametersggg
J=2*np.pi*0.015 #could be made bigger eventually
nu=2*np.pi*4 #change Energies so this is the resonator's
# delta=2*np.pi*0.1 #change Energies so this is really the dfetuning
delta=0
# de=-1*delta
de=-1
wq=2*np.pi*5
EJ=2*np.pi*12
eta=.1
phiq=0.43
phia=0.019
kappa= 0.002
ct=-2*np.pi/-0.09725 #compensating tone: 2pi over the period of the rabi oscillation of empty transmon when driven resonantly


n=np.diag(np.arange(nmax))
a=np.diag(np.sqrt(np.arange(1,nmax)),1)
x=(a+a.T)
idv=np.eye(nmax)

En=np.array([wq, nu+delta, wq+de, nu+delta],dtype=complex)
#En=np.ones(sites)

H0=0

for i in np.arange(sites):
    HN=1
    for j in np.arange(sites):
        if i==j:
            HN=np.kron(HN,n)
        else:
            HN=np.kron(HN,idv)
    H0=H0+En[i]*HN

idv2=np.kron(idv,idv)
xq=phiq*np.kron(x,idv)
xr=phia*np.kron(idv,x)
ID=np.kron(idv2,idv2)

nv0=np.kron(np.kron(idv,n),idv2)
nv1=np.kron(idv2,np.kron(idv,n))

kav0=kappa*np.kron(np.kron(idv,a),idv2)
kav1=kappa*np.kron(idv2,np.kron(idv,a))
adv0=np.kron(np.kron(idv,a.T),idv2)
adv1=np.kron(idv2,np.kron(idv,a.T))


knsum=kappa*(nv0+nv1)/2

Xsum1=np.kron(xq+xr,idv2)
EX1=expm(-1j*Xsum1)
EX1d=EX1.conj().T

Xsum2=np.kron(idv2,xq+xr)
EX2=expm(-1j*Xsum2)
EX2d=EX2.conj().T

EX=EJ*(EX1+EX2)/2
EXd=EJ*(EX1d+EX2d)/2

Xqsum=(np.kron(xq,idv2)/phiq+np.kron(idv2,xq)/phiq)

ctXrsum=ct*(np.kron(xr,idv2)/phia+np.kron(idv2,xr)/phia)

eta_ct_Xsum=ctXrsum+Xqsum
EJXsumSQ=EJ*Xsum1@Xsum1/2+EJ*Xsum2@Xsum2/2

JCoupl=J*np.kron(xq,xq)/phiq**2
H0=H0+JCoupl+EJXsumSQ+EX+EXd


def Lrho(rho,t):
    etat=eta*np.cos(nu*t)
    etatXsum=etat*eta_ct_Xsum
    drhodt = (-1j*(H0+etatXsum)-knsum)@rho
    drhodt = drhodt + drhodt.conj().T + kav0@rho@adv0 + kav1@rho@adv1
    return drhodt

Psi0=np.zeros(nmax)
Psi0[0]=1
Psi1=np.zeros(nmax)
Psi1[1]=1
Psi=np.kron(Psi1,np.kron(Psi0,np.kron(Psi0,Psi0)))

rho=np.outer(Psi,Psi.conj().T)

n0=np.kron(n,np.kron(idv,idv2))
n1=np.kron(idv2,np.kron(n,idv))

#Propagation with Runge Kutta
njt=np.zeros((tsteps,4))
with tqdm(total=10000) as pbar:
    for i in np.arange(tsteps):
        njt[i,0]=np.abs(np.trace(n0@rho))
        njt[i,1]=np.abs(np.trace(n1@rho))
        njt[i,2]=np.abs(np.trace(nv0@rho))
        njt[i,3]=np.abs(np.trace(nv1@rho))
        k1 = Lrho(rho,i*dt)
        k2 = Lrho(rho+dt*k1/2,(i+1/2)*dt)
        k3 = Lrho(rho+dt*k2/2,(i+1/2)*dt)
        k4 = Lrho(rho+dt*k3,(i+1)*dt)
        rho = rho +dt*(k1+2*k2+2*k3+k4)/6
        pbar.update(1)

np.save("TransmonSim_kappa_{}.npy".format(kappa), njt)