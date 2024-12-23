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
    return qnew_arr
'''
#Quick Testing        
total = wig.DW(2)
frame = total[:4]
dual = total[4:]

state = np.array([[1, 1], [1, 1]])
s = QPRs(state/2, frame)

meas = np.array([[1, 0], [0, 0]])
m = QPRm(meas, dual)
born = Born(m, s)

meas1 = np.array([[0, 0], [0, 1]])
m1 = QPRm(meas1, dual)


unitary = (1/np.sqrt(2)) * np.array([[1,1], [1,-1]])
u = QPRu(unitary, frame, dual)
evo = Evo(u, s) 

FG = wig.DW(3)
frame = FG[:9]
dual = FG[9:]
omega = np.exp(complex(0, 2*np.pi/3))
Hadamard = complex(0, -1/np.sqrt(3)) * np.array([[1,1,1],
                                                     [1, omega, omega**2],
                                                     [1, omega**2, omega]])
Phase = (omega**(8/3)) * np.diag([1,1,omega])
test = QPRu(Phase, frame, dual)
'''
