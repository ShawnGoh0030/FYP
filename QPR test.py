import sympy as syp
import sympy.physics.quantum as qm

dim = 2 #dimensionality

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

    temp = D(dim) #List of displacement operators D

    A0 = syp.zeros(dim, dim)
    for entry in temp:
        A0 += (entry /dim)
 
    F = []
    for i in range(dim**2): 
        #Phase space point operators A
        A = temp[i] * A0 * qm.dagger.Dagger(temp[i])
        F.append(A/dim)

    return F #Discrete Wigner dual operator G_i = dim*F_i

'''
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

print(Frame()) # 1 2 3 4
print()        # Maps to
print(DWF(2))  # 1 3 2 4
'''

#Calculates density matrix of a bloch vector
def bloch_to_dm(bloch):

    dmatrix = (1/2) * (syp.eye(2) + bloch[0]*X(2) +
                       syp.I*bloch[1]*X(2)*Z(2) + bloch[2]*Z(2))
    return dmatrix

#Calculates density matrix of a state
def state_to_dm(state):

    density_matrix = state * qm.dagger.Dagger(state)
    return density_matrix

#2D SICPOVM
def SICPOVM_2():

    frame = []
    vectors = [ [1/syp.sqrt(3), 1/syp.sqrt(3), 1/syp.sqrt(3)],
                [1/syp.sqrt(3), 1/syp.sqrt(3), -1/syp.sqrt(3)],
                [-1/syp.sqrt(3), 1/syp.sqrt(3), 1/syp.sqrt(3)],
                [-1/syp.sqrt(3), -1/syp.sqrt(3), -1/syp.sqrt(3)] ]

    for entry in vectors:
        frame.append((1/2) * bloch_to_dm(entry))

    return frame

#QPR representation
def QPR(density_matrix, frame):

    qpr = []
    exact = []
    numerical = []
    for entry in frame:
        temp = syp.Trace(density_matrix * entry)
        temp2 = syp.simplify(temp)      #simplifies experession
        exact.append(temp2)             #list of exact expressions
        numerical.append(temp2.evalf()) #list of numerical values
    qpr = exact + numerical
    return(qpr)

'''
#Testing examples 
# 0, 1, +, +y
example = [syp.Matrix([1, 0]),
           syp.Matrix([0, 1]),
           syp.Matrix([1/syp.sqrt(2), 1/syp.sqrt(2)]),
           syp.Matrix([1/syp.sqrt(2), syp.I/syp.sqrt(2)])]
for entry in example:
    test1 = QPR(state_to_dm(entry), DWF(2))[4:]
    print(test1)
    summ = 0
    for item in test1:
        summ += item
    print(summ)
    print()

#-1/sqrt(3) [1,1,1]
bloch = [-1/syp.sqrt(3), -1/syp.sqrt(3), -1/syp.sqrt(3)]
test2 = QPR(bloch_to_dm(bloch), DWF(2))[4:]
print(test2)
summ = 0
for item in test2:
    summ += item
print(summ)
print()
'''
