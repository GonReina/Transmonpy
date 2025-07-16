import numpy as np
import matplotlib.pyplot as plt
import os

# constants
tsteps = 50000
dt = 0.0002
time = np.arange(tsteps) * dt

plt.figure(figsize=(12, 6))

# indices we want to annotate (0-based)
targets = [0.076, 0.129, 0.249, 0.302]
# choose an x‐position (e.g. at 80% of the time axis)
idx_annot = int(0.8 * tsteps)

# extract numeric ct and sort
def ct_from_fn(fn):
    return float(fn.split('_')[-1].replace('.npy', ''))

i = 0
for fn in os.listdir('.'):
    if fn.endswith('.npy'):
        i += 1
        ct = ct_from_fn(fn)
        data = np.load(fn)
        pop3 = data[:, 2]
        plt.plot(time, pop3, label=f"ct={ct:.2f}", lw=1.5)

        # if this curve is one of our targets, annotate
        if ct in targets:
            x0 = time[idx_annot]
            y0 = pop3[idx_annot]
            plt.annotate(
                        f"ct={ct:.2f}",
                        xy=(x0, y0),
                        xytext=(20, 15),                   # offset of text box
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle="->",   # keeps your arrow
                                        lw=1,
                                        color='black'),
                        bbox=dict(
                            boxstyle="round,pad=0.3",      # rounded box
                            fc="white",                    # white fill
                            ec="black",                    # black border
                            alpha=0.9                      # slight transparency
                        ),
                        fontsize=10,
                        color='black',
                        zorder=10                           # draw on top
                    )


# styling
plt.xlabel('Time (s)')
plt.ylabel('Population')
plt.title('Resonator 1 population')

# # custom arrow for legend replacement
# plt.arrow(0.8, 0.01, 0, 0.02, width=0.02,
#           head_width=0.4, head_length=0.002,
#           fc='black', ec='black')
# plt.text(0.7, 0.042, 'Increasing ct values',
#          horizontalalignment='center',
#          verticalalignment='center',
#          fontsize=12)

plt.grid()
plt.savefig('resonator_0_ct_range_plot.png', dpi=300)
plt.show()
