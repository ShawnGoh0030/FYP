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

qubits = 2          #initial number of qubits
max_qubits = 2      #run up until x qubits

p_diff = []
samples_needed = []
qubit_count = []

#Some frequently used objects
QPRu_x = qr.QPRu(wig.X(2), frame, dual)
QPRu_z = qr.QPRu(wig.Z(2), frame, dual)
QPRu_z_ancilla = np.kron( np.eye(16), QPRu_z )

Hadamard = (1/np.sqrt(2)) * np.array([[1,1],[1,-1]])
QPRu_Hadamard = qr.QPRu(Hadamard, frame, dual)
QPRu_2Hadamard = np.kron(QPRu_Hadamard, QPRu_Hadamard)
QPRu_Hadamard_wo_ancilla = np.kron(QPRu_2Hadamard, np.eye(4))
QPRu_Hadamard_w_ancilla = np.kron(QPRu_2Hadamard, QPRu_Hadamard)

def control_seq(sequence):

    control_0 = qr.QPRu(np.diag([1,0]), frame, dual)
    control_1 = qr.QPRu(np.diag([0,1]), frame, dual)
    control = [control_0, control_1]
    result = np.array([1])
    for entry in sequence:
        result = np.kron(result, control[entry])
    return result

#Running Grover Algorithm for varying numbers of qubits
while qubits <= max_qubits:
   
    QPRs_initial = cg.initial(qubits + 1, 0, 2, frame)
    QPRs = QPRs_initial
    QPRu_list = []

    #Step 1 - even superposition by Hadamard
    QPRu_list.append(QPRu_Hadamard_w_ancilla)
    QPRu_list.append(QPRu_z_ancilla)

    #Step 2 - oracle flips sign of target
    target = np.random.choice(2, qubits)
    target = np.array([0,0])
    count = 0
    control = [ [[0],[1]] ]
    while count < qubits - 1:
        temp = []
        for i in range(2):
            for entry in control[count]:
                temp.append(entry + [i])
        control.append(temp)
        count += 1

    QPRu_step2 = np.zeros(( 4**(qubits+1), 4**(qubits+1) ))
    temp = np.zeros((4**qubits, 4**qubits))
    for entry in control[count]:
        if (entry - target).any() == False:
            QPRu_step2 += np.kron(control_seq(entry), QPRu_x)
            temp += control_seq(entry)
        else:
            QPRu_step2 += np.kron(control_seq(entry), np.eye(4))
            temp ++ control_seq(entry)
            
    '''               
    QPRu_flip = qr.QPRu(np.diag([1, -1]), frame, dual)
    QPRu_step2 = np.eye(4)
    count = 1
    if count == qubits - 1:
        QPRu_step2 = np.kron(QPRu_step2, QPRu_flip)
        count += 1
    elif count < qubits:
        QPRu_step2 = np.kron(QPRu_step2, np.eye(4))
         count += 1
    '''
    #Step 3 - diffusor operator
    QPRu_step3 = []
    QPRu_step3.append(QPRu_Hadamard_wo_ancilla)
    anti_control = np.array([0]*qubits)
    QPRu_step3_middle = np.zeros(( 4**(qubits+1), 4**(qubits+1) ))
    for entry in control[count]:
        if (entry - anti_control).any() == False:
            QPRu_step3_middle += np.kron(control_seq(entry), QPRu_x)
            ttt = np.kron(control_seq(entry), QPRu_x)
        else:
            QPRu_step3_middle += np.kron(control_seq(entry), np.eye(4))
    QPRu_step3.append(QPRu_step3_middle)
    QPRu_step3.append(QPRu_Hadamard_wo_ancilla)
    
    #Measurement for target state
    QPRm_0 = qr.QPRm(np.diag([1,0]), dual)
    QPRm_1 = qr.QPRm(np.diag([0,1]), dual)
    QPRm_single = [QPRm_0, QPRm_1]
    if target[0] == 0:
        QPRm_target = QPRm_0
    else:
        QPRm_target = QPRm_1
    count = 1
    while count < qubits:
        QPRm_target = np.kron(QPRm_target, QPRm_single[target[count]])
        count += 1
    '''
    QPRm_single = qr.QPRm(np.diag([0,1]), dual)
    QPRm_target = QPRm_single
    count = 1
    while count < qubits:
        QPRm_target = np.kron(QPRm_target, QPRm_single)
        count += 1
    '''
    
    #Running the circuit
    Upper_Bound = np.floor((np.pi/4) * np.sqrt(2**qubits))
    count = 0
    while count < Upper_Bound:
        QPRu_list.append(QPRu_step2)
        QPRu_list += QPRu_step3 
        count += 1
    for entry in QPRu_list:  
        QPRs = qr.Evo(entry, QPRs)
    
    #Removing ancilla
    QPRs_removed = []
    count = 0
    summ = 0
    for entry in QPRs:
        summ += entry
        count += 1
        if count == 4:
            QPRs_removed.append(summ)
            summ = 0
            count = 0 
      
    Born = qr.Born(QPRm_target, QPRs_removed)
    print(target)
    print(Born)
    qubits += 1
    
    #Results
    2 - 0.9999999999999987
    3 - 0.9453124999999978
    4 - 0.9613189697265594
    5 - 0.9991823155432897
    
    #Sampling - slow
    Data = sm.new_sampling(QPRu_list, QPRs_initial, QPRm_target)
    print(Data)
    p_diff.append(abs(Born - Data[0]))
    samples_needed.append(Data[1])
    print(qubits,'qubits,',Data[1],'samples in',Data[2])
    qubit_count.append(qubits)
    
    #Calculates samples needed
    negtot = ne.negs(QPRs_initial)      #negativity of initial state
    for QPRu in QPRu_list:              #multiples by negativity of each s
        negtot *= max(ne.negu(QPRu))
    absm = []
    for entry in QPRm_target:           #list of absolute values of m
        absm.append(abs(entry))
    negtot *= max(absm)
    samples_needed.append(negtot)
    qubit_count.append(qubits)
    
    QPRu_Hadamard_wo_ancilla = np.kron(QPRu_Hadamard, QPRu_Hadamard_wo_ancilla)
    QPRu_Hadamard_w_ancilla = np.kron(QPRu_Hadamard, QPRu_Hadamard_w_ancilla)
    QPRu_z_ancilla = np.kron(np.eye(4),QPRu_z_ancilla)
    qubits += 1
'''
#Plotting
plt.scatter(qubit_count, samples_needed)
plt.xlabel("Number of qubits")
plt.ylabel("Samples needed")
plt.yscale('log', base =10)
plt.show()

state0 = np.diag([1,0])
state1 = np.diag([0,1])

test00 = np.kron( np.kron(state0, state0), wig.X(2) )
test01 = np.kron( np.kron(state0, state1), np.eye(2) )
test10 = np.kron( np.kron(state1, state0), np.eye(2) )
test11 = np.kron( np.kron(state1, state1), np.eye(2) )
testtot = test00 + test01 + test10 + test11
FG8 = wig.DW(8)
frame8 = FG8[:64]
dual8 = FG8[64:]
test00 = np.kron(state0, state0)
test01 = np.kron(state0, state1)
test10 = np.kron(state1, state0)
test11 = np.kron(state1, state1)
testtot = test00 + test01 + test10 + test11
FG4 = wig.DW(4)
frame4 = FG4[:16]
dual4 = FG4[16:]
testQPR = qr.QPRu(testtot, frame4, dual4)
#print(qr.QPRu(test10,frame8,dual8) - ttt)
test = control_seq((0,1))
con0 = qr.QPRu(state0, frame, dual)
con1 = qr.QPRu(state1, frame, dual)
H3 = np.array([[1,1,1],
               [1, np.exp(2j*np.pi/3), np.exp(4j*np.pi/3)],
               [1, np.exp(4j*np.pi/3), np.exp(2j*np.pi/3)]])/np.sqrt(3)

FG8 = wig.DW(8)
frame8 = FG8[:64]
dual8 = FG8[64:]
I_H = qr.QPRu(np.kron(np.eye(2),Hadamard), frame4, dual4)
test = np.kron(np.eye(4), QPRu_Hadamard)
IH = np.kron(np.eye(2),Hadamard)
'''
