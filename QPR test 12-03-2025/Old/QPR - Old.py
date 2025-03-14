import Wigner as wig
import numpy as np

#QPR representation of state
def QPRs(density_matrix, frame):

    qpr = []
    for entry in frame:
        temp = np.trace(density_matrix @ entry)
        qpr.append(temp)
    return(qpr)

#QPR representation of measure
def QPRm(measure, dual):

    m = []
    for entry in dual:
        temp = np.trace(measure @ entry)
        m.append(temp)
    return(m)

#Born rule in QPR
def Born(QPRm, QPRs):

    summ = 0
    for i in range(len(QPRm)):
        temp = QPRm[i] * QPRs[i]
        summ += temp
    return summ

#QPR representation of unitary (S matrix)
def QPRu(unitary, frame, dual):

    size = len(dual)            #Equivalent to dim**2
    s = []                      #S_ij
    for i in range(size):
        row = []                #i-th row of S_ij
        for j in range(size):
            temp = frame[i] @ unitary @ dual[j] @ np.conjugate(np.transpose(unitary))
            row.append(np.trace(temp)) 
        s.append(row)
    s_arr = np.array(s)
    tol = 1e-10                 #Setting negligible values to 0
    s_arr.real[abs(s_arr.real) < tol] = 0.0
    s_arr.imag[abs(s_arr.imag) < tol] = 0.0
    return s_arr

#Evolution of state from unitary in QPR
def Evo(QPRu, QPRs):

    qnew = QPRu @ QPRs
    '''
    qnew = []
    for i in range(len(QPRs)):
        summ = 0
        for j in range(len(QPRs)):
            summ += QPRu[i][j] * QPRs[j]
        qnew.append(summ)
    qnew_arr = np.array(qnew)
    tol = 1e-10                 #Setting negligible values to 0
    qnew_arr.real[abs(qnew_arr.real) < tol] = 0.0
    qnew_arr.imag[abs(qnew_arr.imag) < tol] = 0.0
    '''
    return qnew
'''
#Quick Testing        
total = wig.DW(2)
frame = total[:4]
dual = total[4:]

Hadamard = (1/np.sqrt(2)) * np.array([[1,1], [1,-1]])
QPRu_Hadamard = QPRu(Hadamard, frame, dual)
print(QPRu_Hadamard)
test = frame[0] @ Hadamard @ dual[3] @ Hadamard
Y = np.array([[0, complex(0, -1)],[complex(0,1),0]])
frame_alt = []
for i in range(2):
    for j in range(2):
        temp = np.eye(2) + ((-1)**i)*wig.Z(2) + ((-1)**j)*wig.X(2) + ((-1)**(i+j))*Y
        temp2 = temp/4
        frame_alt.append(temp2)
test2 = frame_alt[0] @ Hadamard @ dual[3] @ Hadamard
'''
