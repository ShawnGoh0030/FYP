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

#Set-up
FG = wig.DW(3)
frame3 = FG[:9]
dual3 = FG[9:]

def nextframe(frame):
    new = []
    for f1 in frame:
        for f2 in frame3:
            new.append( np.kron(f1, f2) )
    return new

def nextdual(dual):
    new = []
    for g1 in dual:
        for g2 in dual3:
            new.append( np.kron(g1, g2) )
    return new

qutrits = 3          #initial number of qubits
max_qutrits = 3      #run up until x qubits

p_diff = []
samples_needed = []
qubit_count = []

#Some frequently used operators
frame = nextframe(frame3)
dual = nextdual(dual3)

Z = wig.Z(3)
Z_dagger = np.transpose( np.conjugate(Z) )
mult_Z = np.kron(Z, Z)
mult_Z_dagger = np.kron(Z_dagger, Z_dagger)

omega = np.exp(2j*np.pi/3)
Hadamard = (1/np.sqrt(3)) * np.array([[1,1,1],
                                      [1,omega,omega**2],
                                      [1,omega**2,omega]])
Hadamard_dagger = np.transpose( np.conjugate(Hadamard) )
mult_H = np.kron(Hadamard, Hadamard)
mult_H_dagger = np.kron(Hadamard_dagger, Hadamard_dagger)

count = 2
while count < qutrits:
    frame = nextframe(frame)
    dual = nextdual(dual)
    mult_Z = np.kron(mult_Z, Z)
    mult_Z_dagger = np.kron(mult_Z_dagger, Z_dagger)
    mult_H = np.kron(mult_H, Hadamard)
    mult_H_dagger = np.kron(mult_H_dagger, Hadamard_dagger)
    count += 1
    

def control_seq(sequence):

    #control_0 = qr.QPRu(np.diag([1,0,0]), frame, dual)
    #control_1 = qr.QPRu(np.diag([0,1,0]), frame, dual)
    #control_2 = qr.QPRu(np.diag([0,0,1]), frame, dual)
    #control = [control_0, control_1, control_2]
    result = np.array([1])
    normal = [np.diag([1,0,0]), np.diag([0,1,0]), np.diag([0,0,1])]
    temp = np.array([1])
    for entry in sequence:
        #result = np.kron(result, control[entry])
        temp = np.kron(temp, normal[entry])
    return temp

#Running Grover Algorithm for varying numbers of qutrits
while qutrits <= max_qutrits:

    state = [1] + [0]*(3**qutrits - 1)
    dm = np.diag(state)
    QPRs_initial = qr.QPRs(dm ,frame)
    QPRs = QPRs_initial
    QPRu_list = []

    #Step 1 - even superposition by Hadamard
    QPRu_mult_H = qr.QPRu(mult_H, frame, dual)
    QPRu_list.append(QPRu_mult_H)

    #Step 2 - oracle flips sign of target
    target = np.random.choice(3, qutrits)
    #target = np.array([0,0])
    count = 0
    control = [ [[0],[1],[2]] ]
    while count < qutrits - 1:
        temp = []
        for i in range(3):
            for entry in control[count]:
                temp.append(entry + [i])
        control.append(temp)
        count += 1
    
    #QPRu_step2 = np.zeros(( 9**qutrits, 9**qutrits ))
    oracle = np.zeros(( 3**qutrits, 3**qutrits ))
    for entry in control[count]:
        if (entry - target).any() == False:
            oracle += -control_seq(entry)

        else:
            oracle += control_seq(entry)
    QPRu_step2 = qr.QPRu(oracle, frame, dual)

    #Step 3 - diffusor operator
    QPRu_step3 = []
    QPRu_step3.append(QPRu_mult_H)
    QPRu_step3.append(qr.QPRu(mult_Z, frame, dual))

    anti_control = np.array([2]*qutrits)
    QPRu_step3_middle = np.zeros(( 3**qutrits, 3**qutrits ))
    for entry in control[count]:
        if (entry - anti_control).any() == False:
            QPRu_step3_middle += -control_seq(entry)
        
        else:
            QPRu_step3_middle += control_seq(entry)
    QPRu_step3.append(qr.QPRu(QPRu_step3_middle, frame, dual))

    QPRu_step3.append(qr.QPRu(mult_Z_dagger,frame,dual))
    QPRu_step3.append(qr.QPRu(mult_H_dagger,frame,dual))
    
    #Measurement for target state
    M0 = np.diag([1,0,0])
    M1 = np.diag([0,1,0])
    M2 = np.diag([0,0,1])
    M_single = [M0, M1, M2]
    M_target = M_single[target[0]]
    count = 1
    while count < qutrits:
        M_target = np.kron(M_target, M_single[target[count]])
        count += 1
    QPRm_target = qr.QPRm(M_target, dual)
    
    #Running the circuit
    Upper_Bound = np.floor((np.pi/4) * np.sqrt(2**qutrits))
    count = 0
    while count < Upper_Bound:
        QPRu_list.append(QPRu_step2)
        QPRu_list += QPRu_step3 
        count += 1
    for entry in QPRu_list:  
        QPRs = qr.Evo(entry, QPRs)
    '''
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
    '''  
    Born = qr.Born(QPRm_target, QPRs)
    print(target)
    print(Born)
    qutrits += 1
    print(len(QPRu_list))
    #Results
    #2 - 0.9999999999999987
    #3 - 0.9453124999999978
    #4 - 0.9613189697265594
    #5 - 0.9991823155432897
    
    #Sampling - slow
    Data = sm.new_sampling(QPRu_list, QPRs_initial, QPRm_target)
    print(Data)
    p_diff.append(abs(Born - Data[0]))
    print(p_diff)
    samples_needed.append(Data[1])
    print(qutrits,'qutrits,',Data[1],'samples in',Data[2])
    qubit_count.append(qutrits)
    
    #Calculates samples needed
    negtot = ne.negs(QPRs_initial)      #negativity of initial state
    for QPRu in QPRu_list:              #multiples by negativity of each s
        negtot *= max(ne.negu(QPRu))
    absm = []
    for entry in QPRm_target:           #list of absolute values of m
        absm.append(abs(entry))
    negtot *= max(absm)
    samples_needed.append(negtot)
    
    #Higher qudits
    if qutrits < max_qutrits:
        mult_Z = np.kron(mult_Z, Z)
        mult_Z_dagger = np.kron(mult_Z_dagger, Z_dagger)
        mult_H = np.kron(mult_H, Hadamard)
        mult_H_dagger = np.kron(mult_H_dagger, Hadamard_dagger)
        frame = nextframe(frame)
        dual = nextdual(dual)  
        qutrits += 1
'''
#Plotting
plt.scatter(qubit_count, samples_needed)
plt.xlabel("Number of qubits")
plt.ylabel("Samples needed")
plt.yscale('log', base =10)
plt.show()
'''
state0 = np.diag([1,0])
state1 = np.diag([0,1])
'''
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
               [1, np.exp(2j*np.pi/3), np.exp(-2j*np.pi/3)],
               [1, np.exp(-2j*np.pi/3), np.exp(2j*np.pi/3)]])/np.sqrt(3)
'''
