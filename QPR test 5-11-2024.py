import sympy as syp
import sympy.physics.quantum as qm
import random

dim = 2   #dimensionality
depth = 2 #length of random circuit
total = 2 #number of initial qudit states

#Shift operator X
def X(dim):

    X = syp.eye(dim)   #Identity matrix 
    lastrow = X.row(dim - 1)

    X.row_del(dim - 1) #Shifting last row to top
    X = X.row_insert(0, lastrow)
    return X

#Phase operator Z
def Z(dim):

    roots = [] #Roots of unity
    for n in range(dim):
        roots.append(syp.exp(2*syp.pi*syp.I*n / dim))
    
    Z = [syp.Matrix()]
    for i in range(dim):
        temp = Z[i]
        Z.append(syp.diag(temp, roots[i]))
    return Z[dim]

#Displacement operators D
def D(dim):

    try:
        exp = syp.mod_inverse(2, dim)
    except:
        exp = 1/2

    omega = syp.exp(2*syp.pi*syp.I / dim)
    D = []
    for z in range(dim):
        for x in range(dim):
            D.append(omega**(-exp*z*x) * (Z(dim)**z) * (X(dim)**x))
    return(D)

#Discrete Wigner frame operators
def DWF(dim):

    #List of displacement operators D
    temp = D(dim) 

    A0 = syp.zeros(dim, dim)
    for entry in temp:
        A0 += (entry /dim)
 
    F = []
    for i in range(dim**2): 
        #Phase space point operators A
        A = temp[i] * A0 * qm.dagger.Dagger(temp[i])
        F.append(A/dim)

    G = [] #Discrete Wigner dual operator G_i = dim*F_i
    for entry in F:
        G.append(entry * dim)
        
    return (F + G) #list of frame and dual

'''
#Legacy content --- not used anymore
#Pauli matrices
sigma1 = syp.Matrix([[0, 1],
                     [1, 0]])

sigma2 = syp.Matrix([[0, -syp.I],
                     [syp.I, 0]])

sigma3 = syp.diag(1, -1)

#Frame
def Frame():

    frame = []
    for i in range(2):
        for j in range(2):
            frame.append( (1/4) * (syp.eye(2) + ((-1)**i)*sigma3 + ((-1)**j)*sigma1 + ((-1)**(i+j))*sigma2))
    return frame

print(Frame())     # 1 2 3 4
print()            # Maps to
print(DWF(2)[:4])  # 1 3 2 4
'''

#Calculates density matrix of a bloch vector
def bloch_to_dm(bloch):

    dmatrix = (1/2) * (syp.eye(2) + bloch[0]*X(2) +
                       bloch[1]*syp.I*X(2)*Z(2) + bloch[2]*Z(2))
    return dmatrix

#Calculates density matrix of a state
def state_to_dm(state):

    density_matrix = state * qm.dagger.Dagger(state)
    return density_matrix

#2D SICPOVM
def SICPOVM_2():

    F = [] #Frame operators
    vectors = [ [1/syp.sqrt(3), 1/syp.sqrt(3), 1/syp.sqrt(3)],
                [1/syp.sqrt(3), 1/syp.sqrt(3), -1/syp.sqrt(3)],
                [-1/syp.sqrt(3), 1/syp.sqrt(3), 1/syp.sqrt(3)],
                [-1/syp.sqrt(3), -1/syp.sqrt(3), -1/syp.sqrt(3)] ]

    for entry in vectors:
        F.append((1/2) * bloch_to_dm(entry))

    G = [] #Dual operators
    for entry in F:
        G.append(6 * entry - syp.eye(2))

    return (F + G)

#QPR representation of state
def QPRs(density_matrix, frame):

    qpr = []
    for entry in frame:
        temp = syp.Trace(density_matrix * entry)
        temp2 = syp.simplify(temp)      #simplifies experession
        qpr.append(temp2.evalf())       #list of numerical values
    return(qpr)

'''
#Testing examples
# 0, 1, +, +y
example = [syp.Matrix([1, 0]),
           syp.Matrix([0, 1]),
           syp.Matrix([1/syp.sqrt(2), 1/syp.sqrt(2)]),
           syp.Matrix([1/syp.sqrt(2), syp.I/syp.sqrt(2)])]
for entry in example:
    test1 = QPRs(state_to_dm(entry), DWF(2)[:4])
    print(test1)
    summ = 0
    for item in test1:
        summ += item
    print(summ)
    print()

#-1/sqrt(3) [1,1,1]
bloch = [-1/syp.sqrt(3), -1/syp.sqrt(3), -1/syp.sqrt(3)]
test2 = QPRs(bloch_to_dm(bloch), DWF(2)[:4])
print(test2)
summ = 0
for item in test2:
    summ += item
print(summ)
print()

vectors = [ [1/syp.sqrt(3), 1/syp.sqrt(3), 1/syp.sqrt(3)],
            [1/syp.sqrt(3), 1/syp.sqrt(3), -1/syp.sqrt(3)],
            [-1/syp.sqrt(3), 1/syp.sqrt(3), 1/syp.sqrt(3)],
            [-1/syp.sqrt(3), -1/syp.sqrt(3), -1/syp.sqrt(3)] ]
test1 = bloch_to_dm(vectors[0]) #error here
test2 = bloch_to_dm(vectors[1])
test3 = bloch_to_dm(vectors[2])
test4 = bloch_to_dm(vectors[3])
print(test2)

trace = syp.Trace(test1 * test4)
print(syp.simplify(trace))
'''

#QPR representation of measure
def QPRm(measure, dual):

    m = []
    for entry in dual:
        temp = syp.Trace(measure * entry)
        temp2 = syp.simplify(temp)      #simplifies experession
        m.append(temp2.evalf())         #list of numerical values
    return(m)

#Born rule
def Born(m, qpr, dim):
    
    check = [] #To check individual calculations
    summ = 0
    for i in range(dim**2):
        temp = m[i] * qpr[i]
        check.append(syp.simplify(temp))
        summ += temp
    return summ

'''
#Testing examples
#plus state
plus = (1/syp.sqrt(2))*syp.Matrix([1,1])
QPRplus = QPRs(state_to_dm(plus), DWF(2)[:4])

#statistical ensemble of 50-50 mix of 0 and 1 states
half = syp.Matrix([[1/2, 0], [0, 1/2]])
QPRhalf = QPRs(half, DWF(2)[:4])

#measuring for 0 state
zeromea = syp.Matrix([[1,0], [0,0]])
mQPR = QPRm(zeromea, DWF(2)[4:])

#probability of 0 state
resultplus = Born(mQPR, QPRplus, dim)
resulthalf = Born(mQPR, QPRhalf, dim)
'''

#QPR representation of unitary (S matrix)
def QPRu(unitary, frame, dual):

    s = []
    for i in range(dim**2):     #S_ij
        temp = []               #row of S_ij
        for j in range(dim**2):
            temp2 = frame[i] * unitary * dual[j] * qm.dagger.Dagger(unitary)
            entry = syp.Trace(syp.simplify(temp2))
            temp.append(syp.simplify(entry))
        s.append(temp)
    return s

#Unitary evolution of state
def Evo(s, qpr, dim):

    qnew = [] #q'

    #calculates each entry q'_i in q'
    for i in range(dim**2):
        entry = 0

        #summation over j for S_ij * q_j
        for j in range(dim**2):
            temp = s[i][j] * qpr[j]
            entry += temp

        qnew.append(entry)

    return qnew
  
'''
#Testing example
test = DWF(2)
Frame = test[:4]
Dual = test[4:]

#S matrix of Hadamard
Hadamard = syp.Matrix([[1/syp.sqrt(2), 1/syp.sqrt(2)],
                       [1/syp.sqrt(2),-1/syp.sqrt(2)]])
testu = QPRu(Hadamard, Frame, Dual)
print(testu)
print()

#initial qubit in 0 state
QPR0 = QPRs(syp.Matrix([[1, 0], [0, 0]]), Frame)
print(QPR0)
print()

#run 0 state qubit into hadamard
zero_into_H = Evo(testu, QPR0, dim)
print(zero_into_H)
print()

#check to see if equal to plus state
plus = (1/syp.sqrt(2))*syp.Matrix([1,1])
QPRplus = QPRs(state_to_dm(plus), Frame)
print(QPRplus)
'''

#Circuit input state
def inputs(total):  #not considering magic states yet

    zero = syp.Matrix([1] + [0]*(dim-1)) #zero state qudit
    initial = [zero]
    for i in range(total - 1):
        temp = initial[i]
        temp2 = qm.tensorproduct.TensorProduct(zero, initial[i]) #all zero state
        initial.append(temp2)        
    return initial[total - 1]

#Generates random circuit
def circuit_gen(depth, total):

    step = 0                 
    inputlist = range(total) #list of n input qudits
    order = []               #the gate to be applied to the nth qudit
    while step < depth:
        gate = random.choice(range(3)) # 0 = Hadamard, 1 = Phase, 2 = CNOT
        if gate != 2:
            order.append((gate, random.choice(range(total))))       #(Hadamard/Phase, target) 
        else:
            order.append((gate, random.sample(range(total), k=2))) #(CNOT, (control, target))
        step += 1

    return order

#Calculates the unitaries at each step of the circuit
def circuit_run(order):

    #only considering qubits for now
    Hadamard = syp.Matrix([[1/syp.sqrt(2), 1/syp.sqrt(2)],
                           [1/syp.sqrt(2),-1/syp.sqrt(2)]])
    Phase = Z(2) ** (1/2)
    CNOT_target = syp.Matrix([[0,1],
                              [1,0]])
    gates = [Hadamard, Phase, CNOT_target]

    ulist = []
    for entry in order:
        #Hadamard/Phase
        if entry[0] != 2:
            target = entry[1]

            #step 0
            if target == 0:
                unitary = [gates[entry[0]]]
            else:    
                unitary = [syp.eye(dim)]
                
            step = 1

            #tensor product identity until target is reached
            while step < target:
                temp = unitary[step - 1]
                temp2 = qm.tensorproduct.TensorProduct(temp, syp.eye(dim))
                unitary.append(temp2)
                step += 1

            #places Hadamard/Phase gate at target (if not already at step 0)
            if step == target:
                temp = unitary[step - 1]
                temp2 = qm.tensorproduct.TensorProduct(temp, gates[entry[0]])
                unitary.append(temp2)
                step += 1

            #tensor product the remaining identities
            while step < total:
                temp = unitary[step - 1]
                temp2 = qm.tensorproduct.TensorProduct(temp, syp.eye(dim))
                unitary.append(temp2)
                step += 1

            ulist.append(unitary[step - 1])

        #CNOT
        else:
            target = entry[1][1]

            #step 0
            if target == 0:
                unitary = [gates[2]]
            else:
                unitary = [syp.eye(dim)]
                
            step = 1

            #tensor product identity unitl target is reached (control qudit is left alone)
            while step < target:
                temp = unitary[step - 1]
                temp2 = qm.tensorproduct.TensorProduct(temp, syp.eye(dim))
                unitary.append(temp2)
                step += 1

            #performs operation at target qudit (if not already at step 0)
            temp = unitary[step - 1]
            temp2 = qm.tensorproduct.TensorProduct(temp, gates[2])
            unitary.append(temp2)
            step += 1

            #fill the rest of the diagonal
            while step < total:
                temp = unitary[step - 1]
                temp2 = qm.tensorproduct.TensorProduct(temp, syp.eye(dim))
                unitary.append(temp2)
                step += 1

            ulist.append(unitary[step - 1])

    return ulist

#Circuit measurement for 0 state for each qudit after circuit
def circuit_mea0(ulist):

    #creating measurement operator
    zero_state = syp.Matrix([1] + [0]*(dim-1)) #zero state qudit
    zero_mea = state_to_dm(zero_state)           
    mlist = [zero_mea] #step 0
    step = 1   
    while step < total:
        temp = mlist[step - 1]
        temp2 = qm.tensorproduct.TensorProduct(temp, zero_mea)
        mlist.append(temp2)
        step += 1
    final_mea = mlist[step - 1]
    
    total_dim = dim ** total                #total dimensionality of n qudits
    FG = DWF(total_dim)
    Frame = FG[:total_dim**2]
    Dual = FG[total_dim**2:]
    initial = inputs(total)
    
    qpr = QPRs(state_to_dm(initial), Frame) #initial QPR of all 0\
    
    #calculates unitary evolution after each gate
    for unitary in ulist:
        temp = QPRu(unitary, Frame, Dual)
        print(len(temp)) #=4 error!
        qpr = Evo(QPRu(unitary, Frame, Dual), qpr, total_dim)
    
    #probability of all 0 measurement
    prob =  Born(QPRm(final_mea, Dual), qpr, total_dim)
    return prob

'''
#Testing example
#Phase on first => Hadamard on second
order = [(1, 0), (0, 1)]
ulist = circuit_run(order)
prob = circuit_mea0(ulist)
print(prob)
'''
#2 qubit grover target at (1,1)
oracle = syp.diag(1, 1, 1, -1)

#after hadamard pre oracle
s1 = 1/2*syp.Matrix([1, 1, 1, 1])

#after oracle
s2 = oracle * s1

def circuit_run_grover(order):

    #only considering qubits for now
    Hadamard = syp.Matrix([[1/syp.sqrt(2), 1/syp.sqrt(2)],
                           [1/syp.sqrt(2),-1/syp.sqrt(2)]])
    Phase = X(2)
    CNOT_target = syp.Matrix([[0,1],
                              [1,0]])
    gates = [Hadamard, Phase, CNOT_target]

    ulist = []
    for entry in order:
        #Hadamard/Phase
        if entry[0] != 2:
            target = entry[1]

            #step 0
            if target == 0:
                unitary = [gates[entry[0]]]
            else:    
                unitary = [syp.eye(dim)]
                
            step = 1

            #tensor product identity until target is reached
            while step < target:
                temp = unitary[step - 1]
                temp2 = qm.tensorproduct.TensorProduct(temp, syp.eye(dim))
                unitary.append(temp2)
                step += 1

            #places Hadamard/Phase gate at target (if not already at step 0)
            if step == target:
                temp = unitary[step - 1]
                temp2 = qm.tensorproduct.TensorProduct(temp, gates[entry[0]])
                unitary.append(temp2)
                step += 1

            #tensor product the remaining identities
            while step < total:
                temp = unitary[step - 1]
                temp2 = qm.tensorproduct.TensorProduct(temp, syp.eye(dim))
                unitary.append(temp2)
                step += 1

            ulist.append(unitary[step - 1])

        #CNOT
        else:
            target = entry[1][1]

            #step 0
            if target == 0:
                unitary = [gates[2]]
            else:
                unitary = [syp.eye(dim)]
                
            step = 1

            #tensor product identity unitl target is reached (control qudit is left alone)
            while step < target:
                temp = unitary[step - 1]
                temp2 = qm.tensorproduct.TensorProduct(temp, syp.eye(dim))
                unitary.append(temp2)
                step += 1

            #performs operation at target qudit (if not already at step 0)
            temp = unitary[step - 1]
            temp2 = qm.tensorproduct.TensorProduct(temp, gates[2])
            unitary.append(temp2)
            step += 1

            #fill the rest of the diagonal
            while step < total:
                temp = unitary[step - 1]
                temp2 = qm.tensorproduct.TensorProduct(temp, syp.eye(dim))
                unitary.append(temp2)
                step += 1

            ulist.append(unitary[step - 1])

    return ulist

#grover circuit order
grover = [(0,0), (0,1), (1,0), (1,1), (0,1), (2, (0, 1)), (0,1), (1,0), (1,1), (0,0), (0, 1)]
test = circuit_run_grover(grover)

states = [s2]
for i in range(len(test)):
    states.append(test[i] * states[i])
final = state_to_dm(states[11])
print(final)

measurements = [ state_to_dm(syp.Matrix([1,0,0,0])),
                 state_to_dm(syp.Matrix([0,1,0,0])),
                 state_to_dm(syp.Matrix([0,0,1,0])),
                 state_to_dm(syp.Matrix([0,0,0,1])) ]


prob = []
for measure in measurements:
    temp = syp.Trace(measure * final)
    temp2 = syp.simplify(temp)
    prob.append(temp2.evalf())
print(prob)
