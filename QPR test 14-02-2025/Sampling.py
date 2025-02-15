import Wigner as wig
import QPR as qr
import Negativity as ne
import numpy as np
import random
import math
import time

#sign function
def sign(num):

    if num < 0:
        return -1
    else:
        return 1

#calculates average of estimator p
def sampling(QPRu_list, initial, QPRm):

    ##calculates samples needed
    QPRs = initial
    d2 = len(QPRs)
    #calculates total forward negativity (negtot)
    negtot = ne.negs(QPRs)      #negativity of initial state
    for QPRu in QPRu_list:      #multiples by negativity of each s
        negtot *= max(ne.negu(QPRu))
    absm = []
    for entry in QPRm:          #list of absolute values of m
        absm.append(abs(entry))
    negtot *= max(absm)
    #samples needed, precision 0.01, confidence 95%
    samples_needed = math.ceil(2 * (0.01**-2) * (negtot**2) * np.log(2/0.05))

    ##sampling distribution probability for initial QPRs
    QPRs_neg = ne.negs(QPRs)
    QPRs_prob = []
    for entry in QPRs:
        QPRs_prob.append(abs(entry)/QPRs_neg)

    ##sampling distribution probability for gates QPRu
    #list of point negativities for each gate
    negu_list = []
    for u in QPRu_list:
        negu_list.append(ne.negu(u))
    #transpose to access columns as row
    QPRulist_transposed = []
    for gate in range( len(QPRu_list) ):
        QPRulist_transposed.append(np.transpose(QPRu_list[gate]))
    #calculating distribution for each gate
    QPRu_prob_list = []
    gate_no = 0
    for gate in QPRulist_transposed:
        QPRu_prob = []      #list of distribution for each column
        for column in gate:
            col = 0
            QPRu_prob.append(np.abs(column) / negu_list[gate_no][col])
            col += 1
        QPRu_prob_list.append(QPRu_prob)
        gate_no += 1

    ##estimator p
    start = time.time()
    samples = 0
    ptotal = 0
    while samples < samples_needed:
        #Randomly generating trajectory
        index = [np.random.choice(d2, p = QPRs_prob)]
        for gate in range( len(QPRu_list) ):
            index.append(np.random.choice(d2, p = QPRu_prob_list[gate][index[gate]]))  
        #Calculating p along trajectory
        lambda0 = QPRs[index[0]]                      
        p_traj = ne.negs(QPRs) * sign(lambda0)
        for gate in range( len(QPRu_list) ):
            lambdai = QPRu_list[gate][index[gate + 1]][index[gate]]
            p_traj *= negu_list[gate][index[gate]] * sign(lambdai)
        #Multiply by relevant entry of QPRm
        p_traj *= QPRm[index[-1]]
        #Summing each individual estimate into ptotal
        ptotal += p_traj
        samples += 1

    end = time.time()
    return [ptotal/samples_needed, samples_needed, (end-start)]



'''
#Testing
FG = wig.DW(2)
frame = FG[:4]
dual = FG[4:]
Hadamard = (1/np.sqrt(2)) * np.array([[1,1], [1,-1]])
h = [qr.QPRu(Hadamard, frame, dual)]
s = qr.QPRs(np.array([[1,0],[0,0]]), frame)
m = qr.QPRm(np.array([[1,0],[0,0]]), dual)
test = sampling(h, s, m)
print(test)
'''
