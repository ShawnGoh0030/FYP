import numpy as np
import sympy as syp
import QPR as qr

x, y, z = syp.symbols('x y z')
j = 1j
D0 = np.eye(2)

D1 = np.array([[syp.sin(y), syp.exp(-j*x)*syp.cos(y)],
               [syp.exp(j*x)*syp.cos(y), syp.sin(y)]])
              
D2 = np.array([[syp.cos(y)*syp.cos(z), syp.exp(-j*x)*(syp.sin(y)*syp.sin(z)+j*syp.sin(z))],
               [syp.exp(j*x)*(syp.sin(y)*syp.sin(z)-j*syp.sin(z)), -syp.cos(y)*syp.cos(z)]])

D3 = np.array([[syp.cos(y)*syp.sin(z), syp.exp(-j*x)*(syp.sin(y)*syp.sin(z)+j*syp.cos(z))],
               [syp.exp(j*x)*(syp.sin(y)*syp.sin(z)-j*syp.cos(z)), -syp.cos(y)*syp.sin(z)]])

col1 = [syp.simplify(np.trace(D0@D1)/2), syp.simplify(np.trace(D1@D1)/2),
        syp.simplify(np.trace(D2@D1)/2), syp.simplify(np.trace(D3@D1)/2)]

col2 = [syp.simplify(np.trace(D0@D2)/2), syp.simplify(np.trace(D1@D2)/2),
        syp.simplify(np.trace(D2@D2)/2), syp.simplify(np.trace(D3@D2)/2)]

col3 = [syp.simplify(np.trace(D0@D3)/2), syp.simplify(np.trace(D1@D3)/2),
        syp.simplify(np.trace(D2@D3)/2), syp.simplify(np.trace(D3@D3)/2)]

col1_subs = []
for i in range(4):
    col1_subs.append( col1[i].subs([(x,0),(y,0),(z,0)]) )

col2_subs = []
for i in range(4):
    col2_subs.append( col2[i].subs([(x,0),(y,0),(z,0)]) )

col3_subs = []
for i in range(4):
    col3_subs.append( col3[i].subs([(x,0),(y,0),(z,0)]) )

D1_0 = np.array([[0,1],[1,0]])
D2_0 = np.array([[1,0],[0,-1]])
D3_0 = np.array([[0,j],[-j,0]])
frame = [D0/2, D1_0/2, D2_0/2, D3_0/2]
dual =  [D0, D1_0, D2_0, D3_0]
Hadamard = np.array([[1,1],[1,-1]])/np.sqrt(2)
test = qr.QPRu(Hadamard, frame, dual)
