import numpy as np
from scipy.linalg import expm

#0 state as [1 0], 1 state as [0 1]

#Shift operator X
def X(dim):

    X = np.eye(dim)        #Identity matrix 
    return np.roll(X, dim) #Shift last row to first


#Phase operator Z
def Z(dim):

    roots = []             #Roots of unity
    for n in range(dim):
        roots.append(np.exp(complex(0, 2*n*np.pi/dim)))    
    Z = np.diag(roots)
    tol = 1e-10            #Setting negligible values to 0
    Z.real[abs(Z.real) < tol] = 0.0
    Z.imag[abs(Z.imag) < tol] = 0.0
    return Z

#Displacement operators D_ij
def D(dim):

    Xm = X(dim)
    Zm = Z(dim)

    temp = np.mod(dim, 2)
    if temp == 0:
        power = 1/2
    else:
        power = (dim + 1)/2
    omega = np.exp(complex(0, 2*np.pi/dim))

    D = []
    for i in range(dim):
        for j in range(dim):
            D.append(omega**(-power*i*j) * np.matmul(np.linalg.matrix_power(Zm, i), np.linalg.matrix_power(Xm, j)))
    return D

#Discrete Wigner frame and dual
def DW(dim):

    temp = D(dim)

    #Phase space point operators A_i = G_i Dual
    A0 = 0
    for entry in temp:
        A0 += (1/dim)*entry
    A = []
    tol = 1e-10            #Setting negligible values to 0
    for entry in temp:
        temp2 = entry @ A0 @ np.conjugate(np.transpose(entry))
        temp2.real[abs(temp2.real) < tol] = 0.0
        temp2.imag[abs(temp2.imag) < tol] = 0.0
        A.append(temp2)
    
    #Frame
    F = []
    for entry in A:
        F.append((1/dim)*entry)
    return F+A

#Rotated frame
def rotate(frame, x, y, z):

    dim = int(np.sqrt(len(frame)))
    rotation_x = expm( complex(0, -x/2)*X(dim) )
    rotation_y = expm( y* (X(dim)@Z(dim)) )
    rotation_z = expm( complex(0, -z/2)*Z(dim) )
    rotation_operator = rotation_z @ rotation_y @ rotation_x
    rotation_operator_dagger = np.transpose( np.conjugate(rotation_operator) )
    rotated_frame = []
    for entry in frame:
        rotated_frame.append(rotation_operator @ entry @ rotation_operator_dagger)
    return rotated_frame
