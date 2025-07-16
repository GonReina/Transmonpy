import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time as t
from IPython import display
 
at=t.time()
 
sites=4
nmax=4
tsteps=10000000
dt=.0002
 
J=np.pi/77 #pi over period of rabi oscillation of excited transmon when driven resonantly
nu=10
delta=1
de=-1*delta
wq=100
EJ=100
eta=1
phiq=0.43
phia=0.019
ct=-2*np.pi/18.5 #compensating tone: 2pi over the period of the rabi oscillation of empty transmon when driven resonantly
 
 
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
 
Xsum1=np.kron(xq+xr,idv2)
EX1=expm(-1j*Xsum1)
EX1d=EX1.conj().T
 
Xsum2=np.kron(idv2,xq+xr)
EX2=expm(-1j*Xsum2)
EX2d=EX2.conj().T
 
EX=EJ*(EX1+EX2)/2
EXd=EJ*(EX1d+EX2d)/2
 
Xqsum=np.kron(xq,idv2)/phiq+np.kron(idv2,xq)/phiq
 
Xrsum=np.kron(xr,idv2)/phia+np.kron(idv2,xr)/phia
 
JCoupl=J*np.kron(xq,xq)/phiq**2
H0=H0+JCoupl
 
#etat=0
#H0=(H0+EX*np.exp(-1j*etat)+EXd*np.exp(1j*etat)
#              +EJ*(Xsum1+etat)@(Xsum1+etat)/2
#              +EJ*(Xsum2+etat)@(Xsum2+etat)/2
#              +J*etat*Xqsum)
#drhodt function
def HPsi(Psi,t):
    etat=eta*np.cos(nu*t)
    dPsidt = -1j*(H0+EX*np.exp(-1j*etat)+EXd*np.exp(1j*etat)
              +EJ*(Xsum1+etat*ID)@(Xsum1+etat*ID)/2
              +EJ*(Xsum2+etat*ID)@(Xsum2+etat*ID)/2
              +ct*etat*Xrsum
              +J*etat*Xqsum)@Psi
    #dPsidt = -1j*H0@Psi
    return dPsidt
 
Psi0=np.zeros(nmax)
Psi0[0]=1
Psi1=np.zeros(nmax)
Psi1[1]=1
Psi=np.kron(Psi1,np.kron(Psi0,np.kron(Psi0,Psi0)))
 
n0=np.kron(n,np.kron(idv,idv2))
n1=np.kron(idv2,np.kron(n,idv))
nv0=np.kron(np.kron(idv,n),idv2)
nv1=np.kron(idv2,np.kron(idv,n))
 
#Propagation with Runge Kutta
njt=np.zeros((tsteps,4))
for i in np.arange(tsteps):
    njt[i,0]=np.abs(Psi.conj().T@n0@Psi)
    njt[i,1]=np.abs(Psi.conj().T@n1@Psi)
    njt[i,2]=np.abs(Psi.conj().T@nv0@Psi)
    njt[i,3]=np.abs(Psi.conj().T@nv1@Psi)

    k1 = HPsi(Psi,i*dt)
    k2 = HPsi(Psi+dt*k1/2,(i+1/2)*dt)
    k3 = HPsi(Psi+dt*k2/2,(i+1/2)*dt)
    k4 = HPsi(Psi+dt*k3,(i+1)*dt)
    Psi = Psi +dt*(k1+2*k2+2*k3+k4)/6
    if i%1000==0:
        #print(str(np.floor(i/tsteps*100+10)/10)+" %", end="\r")
        plt.clf()
        plt.plot(dt*np.arange(tsteps),np.real(njt))
        display.display(plt.gcf())
        #plt.savefig("3tilt0_5chi0.pdf")
        display.clear_output(wait=True)
        t.sleep(1.0)
 
    
#plt.imshow(np.real(njt))
#plt.show()
plt.clf()
plt.plot(dt*np.arange(tsteps),np.real(njt))
plt.title(r'Excitation at Transmon 1 - Vibrations at 0 - Tilt -1')
plt.savefig("N4eta1.pdf")
plt.show()
#plt.plot(np.abs(vk)**2)
#plt.show()
#plt.plot(w[:12],'o')
#plt.show()
print(t.time()-at)
