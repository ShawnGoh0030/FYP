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
import time

FG = wig.DW(2)
frame = FG[:4]
dual = FG[4:]

qubits = 2      #initial number of qubits
max_qubits = 6      #run up until x qubits

p_diff = []
samples_needed = []
qubit_count = []

#Running Grover Algorithm for varying numbers of qubits
while qubits <= max_qubits + 1:
   
    QPRs_initial = cg.initial(qubits, 0, 2, frame)
    QPRs = QPRs_initial
    QPRu_list = []

    #Step 1 - even superposition by Hadamard
    Hadamards = []
    count = 0
    while count < qubits:
        Hadamards.append( (0, count) )
        count += 1
    QPRu_Hadamards = cg.circuit_run(Hadamards, qubits, 2, frame, dual)

    #Step 2 - oracle flips sign of target
    #target is fixed to all ones ( 11...11> )
    QPRu_flip = qr.QPRu(np.diag([1, -1]), frame, dual)
    QPRu_step2 = np.eye(4)
    count = 1
    if count == qubits - 1:
        QPRu_step2 = np.kron(QPRu_step2, QPRu_flip)
        count += 1
    elif count < qubits:
        QPRu_step2 = np.kron(QPRu_step2, np.eye(4))
        count += 1
    
    #Step 3 - diffusor operator
    #2 * 11...11><11...11 - identity
    QPRu_step3 = -QPRu_step2
    
    #Measurement for target state
    QPRm_single = qr.QPRm(np.diag([0,1]), dual)
    QPRm_target = QPRm_single
    count = 1
    while count < qubits:
        QPRm_target = np.kron(QPRm_target, QPRm_single)
        count += 1
        
    #Running the circuit
    #Step 1
    for entry in QPRu_Hadamards:
        QPRs = qr.Evo(entry, QPRs)
        QPRu_list.append(entry)
    #Steps 2&3
    Upper_Bound = np.floor((np.pi/4) * np.sqrt(2**qubits))
    count = 0
    while count < Upper_Bound:
        QPRs = qr.Evo(QPRu_step2, QPRs)
        QPRs = qr.Evo(QPRu_step3, QPRs)
        QPRu_list.append(QPRu_step2)
        QPRu_list.append(QPRu_step3)
        count += 1
    #Reversing Hadamards
    for entry in QPRu_Hadamards:
        QPRs = qr.Evo(entry, QPRs)
        QPRu_list.append(entry)
    Born = qr.Born(QPRm_target, QPRs)
    print(Born)
    '''
    Results
    2 - 0.9999999999999987
    3 - 0.9453124999999978
    4 - 0.9613189697265594
    5 - 0.9991823155432897
    ''' 
    #Sampling - very slow
    Data = sm.sampling(QPRu_list, QPRs_initial, QPRm_target)
    p_diff.append(abs(Born - Data[0]))
    samples_needed.append(Data[1])
    print(qubits,'qubits,',Data[1],'samples in',Data[2])
    qubit_count.append(qubits)
    '''
    #Calculates samples needed
    negtot = ne.negs(QPRs_initial)      #negativity of initial state
    for QPRu in QPRu_list:              #multiples by negativity of each s
        negtot *= max(ne.negu(QPRu))
    absm = []
    for entry in QPRm:                  #list of absolute values of m
        absm.append(abs(entry))
    negtot *= max(absm)
    samples_needed.append(negtot)
    qubit_count.append(qubits)
    '''    
    qubits += 1

#Plotting
plt.scatter(qubit_count, samples_needed)
plt.xlabel("Number of qubits")
plt.ylabel("Samples needed")
plt.yscale('log', base =10)
plt.show()




