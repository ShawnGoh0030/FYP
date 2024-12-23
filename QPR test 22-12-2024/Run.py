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
total = 3

#Number of initial magic states
#k = 0 #for individual testing

#Repeat sampling process for each no. of magic states k_repeat times
k_repeat = 5

#Calculating frame and dual for n qutrits
def FG(total):

    #Frame and dual for single qutrits
    FG = wig.DW(3)
    frame_single = FG[:9]
    dual_single = FG[9:]

    #Frame for n qutrits
    step = 1
    total_frame = [frame_single]
    while step < total:
        temp = []
        for frame1 in total_frame[step - 1]:
            for frame2 in frame_single:
                temp.append( np.kron(frame1, frame2) )
        total_frame.append(temp)
        step += 1

    #Dual for n qutrits
    step = 1
    total_dual = [dual_single]
    while step < total:
        temp = []
        for dual1 in total_dual[step - 1]:
            for dual2 in dual_single:
                temp.append( np.kron(dual1, dual2) )
        total_dual.append(temp)
        step += 1

    return total_frame[step - 1] + total_dual[step - 1]

F_G = FG(total)
frame = F_G[:(3**total)**2]
dual = F_G[(3**total)**2:]

#Constructing initial qutrits
initial = cg.initial(total, k, frame)

#Constructing measurement of all qutrits in 0 state
def zero_meas(total):
    meas1 = np.diag([1,0,0])
    meas_total = meas1
    step = 1
    while step < total:
        meas_total = np.kron(meas_total, meas1)
        step += 1
    QPRm = qr.QPRm(meas_total, dual)
    return(QPRm)

QPRm = zero_meas(total)
'''
#order = cg.circuit_gen(depth, total)
order  = [(1, 0), (0, 1), (2, [1, 0])]
print(order)
print()
QPRu_list = cg.circuit_run(order, total, frame, dual)
Born = cg.meas(QPRu_list, initial, total, dual)
print(QPRu_list)
print()
Born_est = sm.sampling(QPRu_list, initial, QPRm)
print(Born)
print()
print(Born_est)
print()
'''
p_diff = []
samples_needed = []
k_list = []
for m in range(total + 1):
    count = 0
    while count < k_repeat:
        initial = cg.initial(total, m, frame)
        order = cg.circuit_gen(depth, total)
        QPRu_list = cg.circuit_run(order, total, frame, dual)
        Born = cg.meas(QPRu_list, initial, total, dual)
        Born_est = sm.sampling(QPRu_list, initial, QPRm)
        p_diff.append(abs(Born - Born_est[0]))
        samples_needed.append(Born_est[1])
        k_list.append(m)
        count += 1
    
custom_cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', ['blue', 'yellow', 'orange', 'red'])
plt.scatter(k_list, p_diff, c = samples_needed, cmap = custom_cmap)
plt.colorbar()
plt.show()

