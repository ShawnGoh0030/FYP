import Wigner as wig
import QPR as qr
import Negativity as ne
import Circuit_gen as cg
import Sampling as sm
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

#Number of gates
depth = 5

#Number of qudits
total = 2

#Repeat sampling process for each no. of magic states x times
k_repeat = 3

#Qubit or Qutrit
dim = 3
if dim == 3:
    qudit = 'Qutrits'
elif dim == 2:
    qudit = 'Qubits'
FG = wig.DW(dim)
frame = FG[:dim**2]
dual = FG[dim**2:]

#Constructing measurement of al 0 state
QPRm = cg.QPRm_0(dim, total, dual)

#Sampling process
p_diff = []
samples_needed = []
k_list = []
print(total,qudit,depth,'Gates')
for m in range(total + 1):
    count = 0
    while count < k_repeat:
        initial = cg.initial(total, m, dim, frame)
        order = cg.circuit_gen(depth, total)
        QPRu_list = cg.circuit_run(order, total, dim, frame, dual)
        Born = cg.Born_circuit(QPRu_list, initial, QPRm)
        Born_est = sm.sampling(QPRu_list, initial, QPRm)
        p_diff.append(abs(Born - Born_est[0]))
        samples_needed.append(Born_est[1])
        k_list.append(m)
        print(m,'magic states,',Born_est[1],'samples in',Born_est[2])
        count += 1

#Plotting
custom_cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', ['blue', 'yellow', 'orange', 'red'])
plt.scatter(k_list, p_diff, c = samples_needed, cmap = custom_cmap)
plt.colorbar(label = 'Samples needed')
plt.xlabel("Number of initial magic states")
plt.ylabel("p - <p'>")
plt.show()

