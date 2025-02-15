import Wigner as wig
import QPR as qr
import numpy as np
import random

#depth = number of steps/gates, total = number of qudits

#Generates random circuit
def circuit_gen(depth, total):

    step = 0
    order = []
    while step < depth:
        gate = random.choice(range(3)) # 0 = Hadamard, 1 = Phase, 2 = CNOT
        if gate != 2:
            order.append((gate, np.random.choice(2)))      #(Hadamard/Phase, target) 
        else:
            order.append((gate, random.sample(range(total), k=2))) #(CNOT, (control, target))
        step += 1
    return order    #e.g. (1,0) - Phase on qudit 0, (2, (2,3)) - CNOT, qudit 2 is control qudit 3 is target

#Generates initial state with k magic-state qutrits and x-k 0-state qutrits
def initial(total, k, dim, frame):

    if dim == 3:
        #Qutrit
        zeta = np.exp(complex(0, 2*np.pi/9))
        magic_state = (1/np.sqrt(3)) * np.array([[1], [zeta], [zeta**8]])
        magic_state_t = np.conjugate(np.transpose(magic_state))
        den_mat = magic_state @ magic_state_t     #density matrix of magic state
        QPRs_magic = qr.QPRs(den_mat, frame)      #QPRs of magic state
        QPRs_0 = qr.QPRs(np.diag([1,0,0]), frame) #QPRs of 0 state

    elif dim == 2:
        #Qubit
        magic_state = (1/np.sqrt(2)) * np.array([[1], [np.exp(complex(0, np.pi/4))]])
        magic_state_t = np.conjugate(np.transpose(magic_state))
        den_mat = magic_state @ magic_state_t     #density matrix of magic state
        QPRs_magic = qr.QPRs(den_mat, frame)      #QPRs of magic state
        QPRs_0 = qr.QPRs(np.diag([1,0]), frame)   #QPRs of 0 state
    
    QPRs_list = []
    count = 0

    if k == 0:
        QPRs_initial = QPRs_0
    else:
        QPRs_initial = QPRs_magic
    count = 1
    
    while count < total:
        if count < k:
            QPRs_initial = np.kron(QPRs_initial, QPRs_magic)
            count += 1
        else:
            QPRs_initial = np.kron(QPRs_initial, QPRs_0)
            count += 1
    return QPRs_initial

#Runs the generated circuit from initial state
def circuit_run(order, total, dim, frame, dual):

    if dim == 3:
        #Qutrit case
        omega = np.exp(complex(0, 2*np.pi/3))
        Hadamard = complex(0, -1/np.sqrt(3)) * np.array([[1,1,1],
                                                         [1, omega, omega**2],
                                                         [1, omega**2, omega]])
        Phase = (omega**(8/3)) * np.diag([1,1,omega]) 
        x = wig.X(3)
        Identity = np.eye(3)
    
    elif dim == 2:
        #Qubit case
        Hadamard = (1/np.sqrt(2)) * np.array([[1, 1],
                                              [1, -1]])
        Phase = np.array([[1, 0],
                          [0, complex(0, 1)]])
        x = wig.X(2)
        Identity = np.eye(2)
        
    QPRu_gates = [qr.QPRu(Hadamard,frame,dual), qr.QPRu(Phase,frame,dual),
                  qr.QPRu(Identity,frame,dual)]

    QPRu_list = []
    for entry in order:
        #Hadamard/Phase
        if entry[0] != 2:
            target = entry[1]
            gate = entry[0]

            if target == 0:
                QPRu = QPRu_gates[gate]
            else:
                QPRu = QPRu_gates[2]
            count = 1
            #print(QPRu)
            while count < total:
                if count == target:
                    QPRu = np.kron(QPRu, QPRu_gates[gate])
                    count += 1
                else:
                    QPRu = np.kron(QPRu, QPRu_gates[2])
                    count += 1
            QPRu_list.append(QPRu)
            
        #CNOT
        else:
            control = entry[1][0]
            target = entry[1][1]
            
            if dim == 3:
                #Qutrit case, split CNOT into 3 cases then sum together
                QPRu_control = [qr.QPRu(np.diag([1,0,0]), frame, dual),
                                qr.QPRu(np.diag([0,1,0]), frame, dual),
                                qr.QPRu(np.diag([0,0,1]), frame, dual)]

                QPRu_target = [QPRu_gates[2],
                               qr.QPRu(x, frame, dual),
                               qr.QPRu(x@x, frame, dual)] 

                step = 0        #initial qutrit
                if control == step: 
                    CNOT_partial0 = QPRu_control[0]  #control is 0
                    CNOT_partial1 = QPRu_control[1]  #control is 1
                    CNOT_partial2 = QPRu_control[2]  #control is 2
                elif target == step:
                    CNOT_partial0 = QPRu_target[0]   #control is 0
                    CNOT_partial1 = QPRu_target[1]   #control is 1
                    CNOT_partial2 = QPRu_target[2]   #control is 2
                else:
                    CNOT_partial0 = QPRu_gates[2]    #else is identity
                    CNOT_partial1 = QPRu_gates[2]
                    CNOT_partial2 = QPRu_gates[2]
            
                step = 1        #second qutrit and beyond
                while step < total:
                    if control == step:
                        CNOT_partial0 = np.kron(CNOT_partial0, QPRu_control[0])
                        CNOT_partial1 = np.kron(CNOT_partial1, QPRu_control[1])
                        CNOT_partial2 = np.kron(CNOT_partial2, QPRu_control[2])
                    elif target == step:
                        CNOT_partial0 = np.kron(CNOT_partial0, QPRu_target[0])
                        CNOT_partial1 = np.kron(CNOT_partial1, QPRu_target[1])
                        CNOT_partial2 = np.kron(CNOT_partial2, QPRu_target[2])
                    else:
                        CNOT_partial0 = np.kron(CNOT_partial0, QPRu_gates[2])
                        CNOT_partial1 = np.kron(CNOT_partial1, QPRu_gates[2])
                        CNOT_partial2 = np.kron(CNOT_partial2, QPRu_gates[2])
                    step += 1
                    
            elif dim == 2:
                #Qutrit case, split CNOT into 3 cases then sum together
                QPRu_control = [qr.QPRu(np.diag([1,0]), frame, dual),
                                qr.QPRu(np.diag([0,1]), frame, dual)]

                QPRu_target = [QPRu_gates[2],
                               qr.QPRu(x, frame, dual)]

                step = 0        #initial qutrit
                if control == step: 
                    CNOT_partial0 = QPRu_control[0]  #control is 0
                    CNOT_partial1 = QPRu_control[1]  #control is 1
                elif target == step:
                    CNOT_partial0 = QPRu_target[0]   #control is 0
                    CNOT_partial1 = QPRu_target[1]   #control is 1
                else:
                    CNOT_partial0 = QPRu_gates[2]    #else is identity
                    CNOT_partial1 = QPRu_gates[2]
            
                step = 1        #second qutrit and beyond
                while step < total:
                    if control == step:
                        CNOT_partial0 = np.kron(CNOT_partial0, QPRu_control[0])
                        CNOT_partial1 = np.kron(CNOT_partial1, QPRu_control[1])
                    elif target == step:
                        CNOT_partial0 = np.kron(CNOT_partial0, QPRu_target[0])
                        CNOT_partial1 = np.kron(CNOT_partial1, QPRu_target[1])
                    else:
                        CNOT_partial0 = np.kron(CNOT_partial0, QPRu_gates[2])
                        CNOT_partial1 = np.kron(CNOT_partial1, QPRu_gates[2])
                    step += 1
                CNOT_partial2 = 0 #unused in qubit case
                
            CNOT = CNOT_partial0 + CNOT_partial1 + CNOT_partial2
            QPRu_list.append(CNOT)
            
    return QPRu_list

#Generates QPRm of all zero measurement
def QPRm_0(dim, total, dual):

    #Measurement effect for 1 qudit
    if dim == 3:
        single = np.diag([1,0,0])
    elif dim == 2:
        single = np.diag([1,0])
    QPRm_single = qr.QPRm(single, dual)

    QPRm_total = QPRm_single
    count = 1
    while count < total:
        QPRm_total = np.kron(QPRm_total, QPRm_single)
        count += 1
    return QPRm_total

#Calculates Born rule of circuit
def Born_circuit(QPRu_list, QPRs_initial, QPRm):

    QPRs = QPRs_initial
    for QPRu in QPRu_list:
        QPRs = qr.Evo(QPRu, QPRs)
    Born = qr.Born(QPRm, QPRs)
    return Born
'''
#Qutrit
FG = wig.DW(3)
frame = FG[:3**2]
dual = FG[3**2:]
'''

#Qubit
FG = wig.DW(2)
frame = FG[:2**2]
dual = FG[2**2:]
Hadamard = (1/np.sqrt(2)) * np.array([[1, 1],
                                      [1, -1]])
'''
total = 1
initial = initial(total, 0, 3, wig.DW(3)[:9])
omega = np.exp(complex(0, 2*np.pi/3))
Hadamard = complex(0, -1/np.sqrt(3)) * np.array([[1,1,1],
                                                 [1, omega, omega**2],
                                                 [1, omega**2, omega]])
temp = Hadamard
print(qr.QPRu(temp, frame, dual))
'''
'''
order = circuit_gen(5,3)
eg = [(2, [1, 0]),
      (1, 2),
      (2, [2, 1]),
      (2, [2, 0]),
      (1, 2)]
QPRu_list = circuit_run(eg, frame, dual)
u = QPRu_list

print(u)
'''
'''
dim = 2
depth = 3
total = 7
test_circuit = [(0, np.int64(1)), (1, np.int64(2)), (2, [2, 1])]
QPRu_list = circuit_run(test_circuit, total, 2, frame, dual)
print(QPRu_list)
'''
