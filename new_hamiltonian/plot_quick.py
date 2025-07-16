import numpy as np
import matplotlib.pyplot as plt

# constants
tsteps = 1000000
dt = 0.0002
time = np.arange(tsteps) * dt

fn = "TransmonSim_kappa_0.002.npy"
data = np.load(fn)
plt.plot(time, data, lw=1.5)

# stylingplt.plot(time, pop3, label=f"ct={ct:.2f}", lw=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Population')
plt.grid()
plt.show()
