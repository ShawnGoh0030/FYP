import numpy as np
import sympy as syp
import QPR as qr
import Wigner as wig
from scipy.linalg import expm
from scipy.optimize import minimize
from tqdm import tqdm
from matplotlib import cm
import matplotlib.pyplot as plt

total = wig.DW(2)
frame = total[:4]
dual = total[4:]
Hadamard = (1/np.sqrt(2)) * np.array([[1,1], [1,-1]])

x, y, z = syp.symbols('x y z')
j = 1j
I = j
def cos(x):
    return np.cos(x)
def sin(x):
    return np.sin(x)
def exp(x):
    return np.exp(x)

rotation_x = syp.exp( -j*x* wig.X(2)/2 )
rotation_y = syp.exp( -j*y* j*wig.X(2)@wig.Z(2)/2 )
rotation_z = syp.exp( -j*z* wig.Z(2)/2)


rot_x = np.array([[syp.cos(x/2), j*syp.sin(x/2)],
                  [j*syp.sin(x/2), syp.cos(x/2)]])
rot_x_dagger = np.array([[syp.cos(x/2), -j*syp.sin(x/2)],
                         [-j*syp.sin(x/2), syp.cos(x/2)]])

rot_y = np.array([[syp.cos(y/2), syp.sin(y/2)],
                  [-syp.sin(y/2), syp.cos(y/2)]])

rot_z = np.array([[syp.exp(j*z/2), 0],
                  [0, syp.exp(-j*z/2)]])
rot_z_dagger = np.array([[syp.exp(-j*z/2), 0],
                         [0, syp.exp(j*z/2)]])

frame_new = []
dual_new = []
for entry in frame:
    frame_new.append(rot_z @ rot_x @ entry @ rot_x_dagger @ rot_z_dagger)
for entry in dual:
    dual_new.append(rot_z @ rot_x @ entry @ rot_x_dagger @ rot_z_dagger)

test0 = frame_new[0]
test1 = frame_new[1]
test2 = frame_new[2]
test3 = frame_new[3]
ftot = test0 + test1 + test2 + test3
#for entry in ftot:
#   print( syp.simplify(entry) )

check0 = syp.simplify(np.trace(dual_new[0]))
check1 = syp.simplify(np.trace(dual_new[1]))
check2 = syp.simplify(np.trace(dual_new[2]))
check3 = syp.simplify(np.trace(dual_new[3]))
'''
QPRu_Hadamard_new = []
for frame in tqdm(frame_new):
    row = []
    for dual in dual_new:
        entry = np.trace(frame @ Hadamard @ dual @ Hadamard) 
        row.append(syp.simplify(entry))
    QPRu_Hadamard_new.append(row)
    print()
test = np.array(QPRu_Hadamard_new)
'''
x = np.arange(0, 2*np.pi, 0.05)
z = np.arange(0, 2*np.pi, 0.05)
x,z = np.meshgrid(x,z)
H00 = (0.1875*I*(cos(2*x) - 1)*exp(4.0*I*z) - 0.1875*I*(cos(2*x) - 1)*exp(6.0*I*z) - 0.125*I*exp(3.0*I*z)*sin(x/2)**4 + 0.25*(1 + I)*exp(3.0*I*z)*sin(x/2)**3*cos(x/2) - 1.0*(0.25 - 0.25*I)*exp(3.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.125*I*exp(3.0*I*z)*cos(x/2)**4 + 0.25*(-1 + I)*exp(4.0*I*z)*sin(x/2)**4 - 0.5*exp(4.0*I*z)*sin(x/2)**3*cos(x/2) - 0.5*exp(4.0*I*z)*sin(x/2)*cos(x/2)**3 + (0.25 + 0.25*I)*exp(4.0*I*z)*cos(x/2)**4 + 0.5*exp(5.0*I*z)*sin(x/2)**3*cos(x/2) - 0.5*exp(5.0*I*z)*sin(x/2)*cos(x/2)**3 - (0.25 + 0.25*I)*exp(6.0*I*z)*sin(x/2)**4 - 0.5*exp(6.0*I*z)*sin(x/2)**3*cos(x/2) - 0.5*exp(6.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.25*(1 - I)*exp(6.0*I*z)*cos(x/2)**4 + 0.125*I*exp(7.0*I*z)*sin(x/2)**4 + 1.0*(0.25 - 0.25*I)*exp(7.0*I*z)*sin(x/2)**3*cos(x/2) - 0.25*(1 + I)*exp(7.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.125*I*exp(7.0*I*z)*cos(x/2)**4)*exp(-5.0*I*z)
H01 = (-0.03125*(cos(2*x) - 1)*exp(3.0*I*z) - 0.1875*I*(cos(2*x) - 1)*exp(4.0*I*z) - 0.0625*(cos(2*x) - 1)*exp(5.0*I*z) + 0.1875*I*(cos(2*x) - 1)*exp(6.0*I*z) - 0.03125*(cos(2*x) - 1)*exp(7.0*I*z) + 0.125*exp(3.0*I*z)*sin(x/2)**4 - 0.25*exp(3.0*I*z)*sin(x/2)**3*cos(x/2) + 0.25*exp(3.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.125*exp(3.0*I*z)*cos(x/2)**4 - 0.25*I*exp(4.0*I*z)*sin(x/2)**4 - 0.25*I*exp(4.0*I*z)*cos(x/2)**4 + 0.25*exp(5.0*I*z)*sin(x/2)**4 - 0.5*exp(5.0*I*z)*sin(x/2)**3*cos(x/2) + 0.5*exp(5.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.25*exp(5.0*I*z)*cos(x/2)**4 + 0.25*I*exp(6.0*I*z)*sin(x/2)**4 + 0.25*I*exp(6.0*I*z)*cos(x/2)**4 + 0.125*exp(7.0*I*z)*sin(x/2)**4 - 0.25*exp(7.0*I*z)*sin(x/2)**3*cos(x/2) + 0.25*exp(7.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.125*exp(7.0*I*z)*cos(x/2)**4)*exp(-5.0*I*z)
H02 = (0.0625*(cos(2*x) - 1)*exp(3.0*I*z) + 0.0625*(cos(2*x) - 1)*exp(7.0*I*z) + 0.125*I*exp(3.0*I*z)*sin(x/2)**4 - 0.125*I*exp(3.0*I*z)*cos(x/2)**4 + 1.0*(0.5 - 1.0*I)*exp(4.0*I*z)*sin(x/2)**3*cos(x/2) + 1.0*(0.5 + 1.0*I)*exp(4.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.5*exp(5.0*I*z)*sin(x/2)**4 + 0.5*exp(5.0*I*z)*cos(x/2)**4 + 1.0*(0.5 + 1.0*I)*exp(6.0*I*z)*sin(x/2)**3*cos(x/2) + 1.0*(0.5 - 1.0*I)*exp(6.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.125*I*exp(7.0*I*z)*sin(x/2)**4 + 0.125*I*exp(7.0*I*z)*cos(x/2)**4)*exp(-5.0*I*z)
H03 = (-0.03125*(cos(2*x) - 1)*exp(3.0*I*z) - 0.03125*(cos(2*x) - 1)*exp(7.0*I*z) - 0.125*exp(3.0*I*z)*sin(x/2)**4 - 0.25*I*exp(3.0*I*z)*sin(x/2)**3*cos(x/2) - 0.25*I*exp(3.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.125*exp(3.0*I*z)*cos(x/2)**4 + 0.25*exp(4.0*I*z)*sin(x/2)**4 + 1.0*I*exp(4.0*I*z)*sin(x/2)**3*cos(x/2) - 1.0*I*exp(4.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.25*exp(4.0*I*z)*cos(x/2)**4 + 0.25*exp(5.0*I*z)*sin(x/2)**4 + 0.375*exp(5.0*I*z)*sin(x)**2 + 0.25*exp(5.0*I*z)*cos(x/2)**4 + 0.25*exp(6.0*I*z)*sin(x/2)**4 - 1.0*I*exp(6.0*I*z)*sin(x/2)**3*cos(x/2) + 1.0*I*exp(6.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.25*exp(6.0*I*z)*cos(x/2)**4 - 0.125*exp(7.0*I*z)*sin(x/2)**4 + 0.25*I*exp(7.0*I*z)*sin(x/2)**3*cos(x/2) + 0.25*I*exp(7.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.125*exp(7.0*I*z)*cos(x/2)**4)*exp(-5.0*I*z)
H10 = (-0.03125*(cos(2*x) - 1)*exp(3.0*I*z) - 0.1875*I*(cos(2*x) - 1)*exp(4.0*I*z) - 0.0625*(cos(2*x) - 1)*exp(5.0*I*z) + 0.1875*I*(cos(2*x) - 1)*exp(6.0*I*z) - 0.03125*(cos(2*x) - 1)*exp(7.0*I*z) + 0.125*exp(3.0*I*z)*sin(x/2)**4 - 0.25*exp(3.0*I*z)*sin(x/2)**3*cos(x/2) + 0.25*exp(3.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.125*exp(3.0*I*z)*cos(x/2)**4 - 0.25*I*exp(4.0*I*z)*sin(x/2)**4 - 0.25*I*exp(4.0*I*z)*cos(x/2)**4 + 0.25*exp(5.0*I*z)*sin(x/2)**4 - 0.5*exp(5.0*I*z)*sin(x/2)**3*cos(x/2) + 0.5*exp(5.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.25*exp(5.0*I*z)*cos(x/2)**4 + 0.25*I*exp(6.0*I*z)*sin(x/2)**4 + 0.25*I*exp(6.0*I*z)*cos(x/2)**4 + 0.125*exp(7.0*I*z)*sin(x/2)**4 - 0.25*exp(7.0*I*z)*sin(x/2)**3*cos(x/2) + 0.25*exp(7.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.125*exp(7.0*I*z)*cos(x/2)**4)*exp(-5.0*I*z)
H11 = (0.1875*I*(cos(2*x) - 1)*exp(4.0*I*z) - 0.1875*I*(cos(2*x) - 1)*exp(6.0*I*z) + 0.125*I*exp(3.0*I*z)*sin(x/2)**4 + 1.0*(0.25 - 0.25*I)*exp(3.0*I*z)*sin(x/2)**3*cos(x/2) - 0.25*(1 + I)*exp(3.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.125*I*exp(3.0*I*z)*cos(x/2)**4 + (0.25 + 0.25*I)*exp(4.0*I*z)*sin(x/2)**4 + 0.5*exp(4.0*I*z)*sin(x/2)**3*cos(x/2) + 0.5*exp(4.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.25*(-1 + I)*exp(4.0*I*z)*cos(x/2)**4 + 0.5*exp(5.0*I*z)*sin(x/2)**3*cos(x/2) - 0.5*exp(5.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.25*(1 - I)*exp(6.0*I*z)*sin(x/2)**4 + 0.5*exp(6.0*I*z)*sin(x/2)**3*cos(x/2) + 0.5*exp(6.0*I*z)*sin(x/2)*cos(x/2)**3 - (0.25 + 0.25*I)*exp(6.0*I*z)*cos(x/2)**4 - 0.125*I*exp(7.0*I*z)*sin(x/2)**4 + 0.25*(1 + I)*exp(7.0*I*z)*sin(x/2)**3*cos(x/2) - 1.0*(0.25 - 0.25*I)*exp(7.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.125*I*exp(7.0*I*z)*cos(x/2)**4)*exp(-5.0*I*z)
H12 = (-0.03125*(cos(2*x) - 1)*exp(3.0*I*z) - 0.1875*(cos(2*x) - 1)*exp(5.0*I*z) - 0.03125*(cos(2*x) - 1)*exp(7.0*I*z) - 0.125*exp(3.0*I*z)*sin(x/2)**4 + 0.25*I*exp(3.0*I*z)*sin(x/2)**3*cos(x/2) + 0.25*I*exp(3.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.125*exp(3.0*I*z)*cos(x/2)**4 - 0.25*exp(4.0*I*z)*sin(x/2)**4 + 1.0*I*exp(4.0*I*z)*sin(x/2)**3*cos(x/2) - 1.0*I*exp(4.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.25*exp(4.0*I*z)*cos(x/2)**4 + 0.25*exp(5.0*I*z)*sin(x/2)**4 + 0.25*exp(5.0*I*z)*cos(x/2)**4 - 0.25*exp(6.0*I*z)*sin(x/2)**4 - 1.0*I*exp(6.0*I*z)*sin(x/2)**3*cos(x/2) + 1.0*I*exp(6.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.25*exp(6.0*I*z)*cos(x/2)**4 - 0.125*exp(7.0*I*z)*sin(x/2)**4 - 0.25*I*exp(7.0*I*z)*sin(x/2)**3*cos(x/2) - 0.25*I*exp(7.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.125*exp(7.0*I*z)*cos(x/2)**4)*exp(-5.0*I*z)
H13 = (0.0625*(cos(2*x) - 1)*exp(3.0*I*z) + 0.0625*(cos(2*x) - 1)*exp(7.0*I*z) - 0.125*I*exp(3.0*I*z)*sin(x/2)**4 + 0.125*I*exp(3.0*I*z)*cos(x/2)**4 - 1.0*(0.5 + 1.0*I)*exp(4.0*I*z)*sin(x/2)**3*cos(x/2) - 1.0*(0.5 - 1.0*I)*exp(4.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.5*exp(5.0*I*z)*sin(x/2)**4 + 0.5*exp(5.0*I*z)*cos(x/2)**4 - 1.0*(0.5 - 1.0*I)*exp(6.0*I*z)*sin(x/2)**3*cos(x/2) - 1.0*(0.5 + 1.0*I)*exp(6.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.125*I*exp(7.0*I*z)*sin(x/2)**4 - 0.125*I*exp(7.0*I*z)*cos(x/2)**4)*exp(-5.0*I*z)
H20 = (0.0625*(cos(2*x) - 1)*exp(3.0*I*z) + 0.0625*(cos(2*x) - 1)*exp(7.0*I*z) + 0.125*I*exp(3.0*I*z)*sin(x/2)**4 - 0.125*I*exp(3.0*I*z)*cos(x/2)**4 + 1.0*(0.5 - 1.0*I)*exp(4.0*I*z)*sin(x/2)**3*cos(x/2) + 1.0*(0.5 + 1.0*I)*exp(4.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.5*exp(5.0*I*z)*sin(x/2)**4 + 0.5*exp(5.0*I*z)*cos(x/2)**4 + 1.0*(0.5 + 1.0*I)*exp(6.0*I*z)*sin(x/2)**3*cos(x/2) + 1.0*(0.5 - 1.0*I)*exp(6.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.125*I*exp(7.0*I*z)*sin(x/2)**4 + 0.125*I*exp(7.0*I*z)*cos(x/2)**4)*exp(-5.0*I*z)
H21 = (-0.03125*(cos(2*x) - 1)*exp(3.0*I*z) - 0.1875*(cos(2*x) - 1)*exp(5.0*I*z) - 0.03125*(cos(2*x) - 1)*exp(7.0*I*z) - 0.125*exp(3.0*I*z)*sin(x/2)**4 + 0.25*I*exp(3.0*I*z)*sin(x/2)**3*cos(x/2) + 0.25*I*exp(3.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.125*exp(3.0*I*z)*cos(x/2)**4 - 0.25*exp(4.0*I*z)*sin(x/2)**4 + 1.0*I*exp(4.0*I*z)*sin(x/2)**3*cos(x/2) - 1.0*I*exp(4.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.25*exp(4.0*I*z)*cos(x/2)**4 + 0.25*exp(5.0*I*z)*sin(x/2)**4 + 0.25*exp(5.0*I*z)*cos(x/2)**4 - 0.25*exp(6.0*I*z)*sin(x/2)**4 - 1.0*I*exp(6.0*I*z)*sin(x/2)**3*cos(x/2) + 1.0*I*exp(6.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.25*exp(6.0*I*z)*cos(x/2)**4 - 0.125*exp(7.0*I*z)*sin(x/2)**4 - 0.25*I*exp(7.0*I*z)*sin(x/2)**3*cos(x/2) - 0.25*I*exp(7.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.125*exp(7.0*I*z)*cos(x/2)**4)*exp(-5.0*I*z)
H22 = (-0.1875*I*(cos(2*x) - 1)*exp(4.0*I*z) + 0.1875*I*(cos(2*x) - 1)*exp(6.0*I*z) - 0.125*I*exp(3.0*I*z)*sin(x/2)**4 - 0.25*(1 + I)*exp(3.0*I*z)*sin(x/2)**3*cos(x/2) + 1.0*(0.25 - 0.25*I)*exp(3.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.125*I*exp(3.0*I*z)*cos(x/2)**4 + 0.25*(1 - I)*exp(4.0*I*z)*sin(x/2)**4 - 0.5*exp(4.0*I*z)*sin(x/2)**3*cos(x/2) - 0.5*exp(4.0*I*z)*sin(x/2)*cos(x/2)**3 - (0.25 + 0.25*I)*exp(4.0*I*z)*cos(x/2)**4 - 0.5*exp(5.0*I*z)*sin(x/2)**3*cos(x/2) + 0.5*exp(5.0*I*z)*sin(x/2)*cos(x/2)**3 + (0.25 + 0.25*I)*exp(6.0*I*z)*sin(x/2)**4 - 0.5*exp(6.0*I*z)*sin(x/2)**3*cos(x/2) - 0.5*exp(6.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.25*(-1 + I)*exp(6.0*I*z)*cos(x/2)**4 + 0.125*I*exp(7.0*I*z)*sin(x/2)**4 - 1.0*(0.25 - 0.25*I)*exp(7.0*I*z)*sin(x/2)**3*cos(x/2) + 0.25*(1 + I)*exp(7.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.125*I*exp(7.0*I*z)*cos(x/2)**4)*exp(-5.0*I*z)
H23 = (-0.03125*(cos(2*x) - 1)*exp(3.0*I*z) + 0.1875*I*(cos(2*x) - 1)*exp(4.0*I*z) - 0.1875*I*(cos(2*x) - 1)*exp(6.0*I*z) - 0.03125*(cos(2*x) - 1)*exp(7.0*I*z) + 0.125*exp(3.0*I*z)*sin(x/2)**4 + 0.25*exp(3.0*I*z)*sin(x/2)**3*cos(x/2) - 0.25*exp(3.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.125*exp(3.0*I*z)*cos(x/2)**4 + 0.25*I*exp(4.0*I*z)*sin(x/2)**4 + 0.25*I*exp(4.0*I*z)*cos(x/2)**4 + 0.25*exp(5.0*I*z)*sin(x/2)**4 + 0.5*exp(5.0*I*z)*sin(x/2)**3*cos(x/2) - 0.5*exp(5.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.125*exp(5.0*I*z)*sin(x)**2 + 0.25*exp(5.0*I*z)*cos(x/2)**4 - 0.25*I*exp(6.0*I*z)*sin(x/2)**4 - 0.25*I*exp(6.0*I*z)*cos(x/2)**4 + 0.125*exp(7.0*I*z)*sin(x/2)**4 + 0.25*exp(7.0*I*z)*sin(x/2)**3*cos(x/2) - 0.25*exp(7.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.125*exp(7.0*I*z)*cos(x/2)**4)*exp(-5.0*I*z)
H30 = (-0.03125*(cos(2*x) - 1)*exp(3.0*I*z) - 0.03125*(cos(2*x) - 1)*exp(7.0*I*z) - 0.125*exp(3.0*I*z)*sin(x/2)**4 - 0.25*I*exp(3.0*I*z)*sin(x/2)**3*cos(x/2) - 0.25*I*exp(3.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.125*exp(3.0*I*z)*cos(x/2)**4 + 0.25*exp(4.0*I*z)*sin(x/2)**4 + 1.0*I*exp(4.0*I*z)*sin(x/2)**3*cos(x/2) - 1.0*I*exp(4.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.25*exp(4.0*I*z)*cos(x/2)**4 + 0.25*exp(5.0*I*z)*sin(x/2)**4 + 0.375*exp(5.0*I*z)*sin(x)**2 + 0.25*exp(5.0*I*z)*cos(x/2)**4 + 0.25*exp(6.0*I*z)*sin(x/2)**4 - 1.0*I*exp(6.0*I*z)*sin(x/2)**3*cos(x/2) + 1.0*I*exp(6.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.25*exp(6.0*I*z)*cos(x/2)**4 - 0.125*exp(7.0*I*z)*sin(x/2)**4 + 0.25*I*exp(7.0*I*z)*sin(x/2)**3*cos(x/2) + 0.25*I*exp(7.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.125*exp(7.0*I*z)*cos(x/2)**4)*exp(-5.0*I*z)
H31 = (0.0625*(cos(2*x) - 1)*exp(3.0*I*z) + 0.0625*(cos(2*x) - 1)*exp(7.0*I*z) - 0.125*I*exp(3.0*I*z)*sin(x/2)**4 + 0.125*I*exp(3.0*I*z)*cos(x/2)**4 - 1.0*(0.5 + 1.0*I)*exp(4.0*I*z)*sin(x/2)**3*cos(x/2) - 1.0*(0.5 - 1.0*I)*exp(4.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.5*exp(5.0*I*z)*sin(x/2)**4 + 0.5*exp(5.0*I*z)*cos(x/2)**4 - 1.0*(0.5 - 1.0*I)*exp(6.0*I*z)*sin(x/2)**3*cos(x/2) - 1.0*(0.5 + 1.0*I)*exp(6.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.125*I*exp(7.0*I*z)*sin(x/2)**4 - 0.125*I*exp(7.0*I*z)*cos(x/2)**4)*exp(-5.0*I*z)
H32 = (-0.03125*(cos(2*x) - 1)*exp(3.0*I*z) + 0.1875*I*(cos(2*x) - 1)*exp(4.0*I*z) - 0.1875*I*(cos(2*x) - 1)*exp(6.0*I*z) - 0.03125*(cos(2*x) - 1)*exp(7.0*I*z) + 0.125*exp(3.0*I*z)*sin(x/2)**4 + 0.25*exp(3.0*I*z)*sin(x/2)**3*cos(x/2) - 0.25*exp(3.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.125*exp(3.0*I*z)*cos(x/2)**4 + 0.25*I*exp(4.0*I*z)*sin(x/2)**4 + 0.25*I*exp(4.0*I*z)*cos(x/2)**4 + 0.25*exp(5.0*I*z)*sin(x/2)**4 + 0.5*exp(5.0*I*z)*sin(x/2)**3*cos(x/2) - 0.5*exp(5.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.125*exp(5.0*I*z)*sin(x)**2 + 0.25*exp(5.0*I*z)*cos(x/2)**4 - 0.25*I*exp(6.0*I*z)*sin(x/2)**4 - 0.25*I*exp(6.0*I*z)*cos(x/2)**4 + 0.125*exp(7.0*I*z)*sin(x/2)**4 + 0.25*exp(7.0*I*z)*sin(x/2)**3*cos(x/2) - 0.25*exp(7.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.125*exp(7.0*I*z)*cos(x/2)**4)*exp(-5.0*I*z)
H33 = (0.1875*I*(cos(2*x) - 1)*exp(6.0*I*z) + 0.125*I*exp(3.0*I*z)*sin(x/2)**4 - 0.25*(1 - I)*exp(3.0*I*z)*sin(x/2)**3*cos(x/2) + 1.0*(0.25 + 0.25*I)*exp(3.0*I*z)*sin(x/2)*cos(x/2)**3 - 0.125*I*exp(3.0*I*z)*cos(x/2)**4 - 0.25*(1 + I)*exp(4.0*I*z)*sin(x/2)**4 + 0.5*exp(4.0*I*z)*sin(x/2)**3*cos(x/2) + 0.5*exp(4.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.375*I*exp(4.0*I*z)*sin(x)**2 + (0.25 - 0.25*I)*exp(4.0*I*z)*cos(x/2)**4 - 0.5*exp(5.0*I*z)*sin(x/2)**3*cos(x/2) + 0.5*exp(5.0*I*z)*sin(x/2)*cos(x/2)**3 + (-0.25 + 0.25*I)*exp(6.0*I*z)*sin(x/2)**4 + 0.5*exp(6.0*I*z)*sin(x/2)**3*cos(x/2) + 0.5*exp(6.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.25*(1 + I)*exp(6.0*I*z)*cos(x/2)**4 - 0.125*I*exp(7.0*I*z)*sin(x/2)**4 - 1.0*(0.25 + 0.25*I)*exp(7.0*I*z)*sin(x/2)**3*cos(x/2) + 0.25*(1 - I)*exp(7.0*I*z)*sin(x/2)*cos(x/2)**3 + 0.125*I*exp(7.0*I*z)*cos(x/2)**4)*exp(-5.0*I*z)
# H01 = H10
#col0 = syp.simplify(abs(H00) + abs(H10))
#col1 = syp.simplify(abs(H01) + abs(H11))
col0 = abs(H00) + abs(H10) + abs(H20) + abs(H30)
col1 = abs(H01) + abs(H11) + abs(H21) + abs(H31)
col2 = abs(H02) + abs(H12) + abs(H22) + abs(H32)
col3 = abs(H03) + abs(H13) + abs(H23) + abs(H33)
neg = []
size = np.shape(col0)
for i in range(size[0]):
    row = []
    for j in range(size[1]):
        row.append(min(col0[i][j], col1[i][j], col2[i][j], col3[i][j]))
    neg.append(row)
neg = np.array(neg)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x, z, neg, cmap=cm.Blues)

ax.set(xlabel = 'x',
       ylabel = 'z',
       zlabel = 'negativity')

plt.show()
for row in neg:
    print(min(row))

