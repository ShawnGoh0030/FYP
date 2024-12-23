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
            order.append((gate, random.choice(range(total))))      #(Hadamard/Phase, target) 
        else:
            order.append((gate, random.sample(range(total), k=2))) #(CNOT, (control, target))
        step += 1
    return order    #e.g. (1,0) - Phase on qudit 0, (2, (2,3)) - CNOT, qudit 2 is control qudit 3 is target

#Calculates the unitaries at each step of circuit
def circuit_run(order, total, frame, dual):
    
    #Qutrit case
    omega = np.exp(complex(0, 2*np.pi/3))
    Hadamard = complex(0, -1/np.sqrt(3)) * np.array([[1,1,1],
                                                     [1, omega, omega**2],
                                                     [1, omega**2, omega]])
    Phase = (omega**(8/3)) * np.diag([1,1,omega]) 
    x = wig.X(3)    #each CNOT will be calculated individually
    
    '''
    #Qubit case
    Hadamard = (1/np.sqrt(2)) * np.array([[1, 1],
                                          [1, -1]])
    Phase = np.array([[1, 0],
                      [0, complex(0, 1)]])
    x = wig.X(2)
    '''
    gates = [Hadamard, Phase]

    ulist = []
    for entry in order:
        #Hadamard/Phase
        if entry[0] != 2:
            target = entry[1]

            #step 0
            if target == 0:
                unitary = [gates[entry[0]]]
            else:    
                unitary = [np.eye(3)]
                
            step = 1

            #tensor product identity until target is reached
            while step < target:
                temp = unitary[step - 1]
                temp2 = np.kron(temp, np.eye(3))
                unitary.append(temp2)
                step += 1

            #places Hadamard/Phase gate at target (if not already at step 0)
            if step == target:
                temp = unitary[step - 1]
                temp2 = np.kron(temp, gates[entry[0]])
                unitary.append(temp2)
                step += 1

            #tensor product the remaining identities
            while step < total:
                temp = unitary[step - 1]
                temp2 = np.kron(temp, np.eye(3))
                unitary.append(temp2)
                step += 1
            ulist.append(unitary[step - 1])
            
        #CNOT
        else:
            control = entry[1][0]
            target = entry[1][1]

            #Qutrit case, split CNOT into 3 cases
            CNOT_partial = []
            step = 0        #initial qutrit
            if control == step: 
                CNOT_partial.append([np.diag([1,0,0]),  #control is 0
                                     np.diag([0,1,0]),  #control is 1
                                     np.diag([0,0,1])]) #control is 2
            elif target == step:
                CNOT_partial.append([np.eye(3),         #control is 0
                                     x,                 #control is 1
                                     x @ x])            #control is 2
            else:
                CNOT_partial.append([np.eye(3),
                                     np.eye(3),
                                     np.eye(3)])
            
            step = 1        #second qutrit and beyond
            while step < total:
                if control == step:
                    temp = [np.kron(CNOT_partial[step - 1][0], np.diag([1,0,0])),
                            np.kron(CNOT_partial[step - 1][1], np.diag([0,1,0])),
                            np.kron(CNOT_partial[step - 1][2], np.diag([0,0,1]))]
                elif target == step:
                    temp = [np.kron(CNOT_partial[step - 1][0], np.eye(3)),
                            np.kron(CNOT_partial[step - 1][1], x),
                            np.kron(CNOT_partial[step - 1][2], (x @ x))]
                else:
                    temp = [np.kron(CNOT_partial[step - 1][0], np.eye(3)),
                            np.kron(CNOT_partial[step - 1][1], np.eye(3)),
                            np.kron(CNOT_partial[step - 1][2], np.eye(3))]
                CNOT_partial.append(temp)
                step += 1

            CNOT = CNOT_partial[step - 1][0] + CNOT_partial[step - 1][1] + CNOT_partial[step - 1][2]
            ulist.append(CNOT)

    #calculates QPR of each unitary
    QPRu_list = []
    for unitary in ulist:
        QPRu_list.append( qr.QPRu(unitary, frame, dual) )
        
    return QPRu_list

#Generates initial state with k magic state qutrits and total-k 0 state qutrits
def initial(total, k, frame):

    #Qutrit
    zeta = omega = np.exp(complex(0, 2*np.pi/9))
    magic_state = (1/np.sqrt(3)) * np.array([[1], [zeta], [zeta**8]])
    magic_state_t = np.conjugate(np.transpose(magic_state))
    den_mat = magic_state @ magic_state_t       #density matrix of magic state

    #Start counting magic states from the top
    if k == 0:
        qudit1 = np.diag([1,0,0])              
    else:
        qudit1 = den_mat
    step = 1
    initial_den_mat = [qudit1]

    while step < k:
        initial_den_mat.append( np.kron(initial_den_mat[step - 1], den_mat) )
        step += 1

    while step < total:
        initial_den_mat.append( np.kron(initial_den_mat[step - 1], np.diag([1, 0, 0])) )
        step += 1
    
    QPRs_initial = qr.QPRs(initial_den_mat[step - 1], frame)
    return(QPRs_initial)

#Calculates Born rule with measurement of all 0 state
def meas(QPRu_list, initial, total, dual):

    #Qutrit
    meas1 = np.diag([1, 0, 0])  #measurement for 1 qutrit
    '''
    #Qubit
    meas1 = np.diag([1, 0])
    '''

    #Calculating measurement
    meas_total = meas1
    step = 1
    while step < total:
        meas_total = np.kron(meas_total, meas1)
        step += 1
    QPRm = qr.QPRm(meas_total, dual)

    QPRs = initial
    for QPRu in QPRu_list:      #calculates evolution of state from each unitary
        QPRs = qr.Evo(QPRu, QPRs)

    Born = qr.Born(QPRm, QPRs)
    return Born
'''
#Qutrit
FG = wig.DW(27)
frame = FG[:27**2]
dual = FG[27**2:]
initial = initial(3, 0, frame) #No magic states, 3 0 states
'''
'''
#Qubit
FG = wig.DW(8)
frame = FG[:8**2]
dual = FG[8**2:]
'''
'''
total = 3
Hadamard = circuit_run([(0,0)], frame, dual) #Hadamard on first qutrit
#test = meas(Hadamard, initial, total, dual)
#print(test)


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
