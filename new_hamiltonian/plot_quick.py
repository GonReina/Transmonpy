import numpy as np
import matplotlib.pyplot as plt
import os
import sys

if __name__ == "__main__":
    directory = "/home/shared/DATA/July/Old_H_new_params/"
    # Find file that ends in npy
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    npy_file = npy_files[0] if npy_files else None
    data = np.load(os.path.join(directory, npy_file))
    tsteps = int(npy_file.split('_')[-1].replace('.npy', ''))
    print(tsteps)
    print(data.shape)
    print(f"Total time steps: {tsteps}")
    dt = 0.0002
    time = np.arange(tsteps) * dt

    # Plot all columns
    plt.plot(time, data[0:tsteps, :])    
    plt.xlabel('Time (s)')
    plt.ylabel('Population')
    plt.grid()
    # Save the plot to data directory
    plt.savefig(os.path.join(directory, 'population_plot.png'))