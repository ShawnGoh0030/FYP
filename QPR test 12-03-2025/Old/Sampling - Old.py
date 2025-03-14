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

#estimator p, QPRu_list is set of QPRu for the circuit
def p(initial, QPRs_prob, QPRu_list, negu_list, QPRm, QPRulist_transposed):

    #Randomly generating trajectory
    QPRs = initial
    d2 = len(QPRs)
    index = [np.random.choice(range(d2), p = QPRs_prob)]
    for gate in range( len(QPRu_list) ):
        '''
        unitary_transpose = np.transpose(QPRu_list[gate])
        column = unitary_transpose[index[gate]]
        '''
        column = QPRulist_transposed[gate][index[gate]]
        index.append(np.random.choice(d2, p = np.abs(column)/negu_list[gate][index[gate]]))

    #Calculating p along trajectory
    lambda0 = QPRs[index[0]]                      
    p_traj = ne.negs(QPRs) * sign(lambda0)
    for gate in range( len(QPRu_list) ):
        lambdai = QPRu_list[gate][index[gate + 1]][index[gate]]
        p_traj *= negu_list[gate][index[gate + 1]] * sign(lambdai)

    #Multiply by relevant entry of QPRm
    p_traj *= QPRm[index[-1]]
    return p_traj

#calculates average of estimator p
def sampling(QPRu_list, initial, QPRm):

    QPRs = initial
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
    print(samples_needed)

    #sampling distributions probability for initial QPRs
    QPRs_neg = ne.negs(QPRs)
    QPRs_prob = []
    for entry in QPRs:
        QPRs_prob.append(abs(entry)/QPRs_neg)

    #list of point negativities for each gate
    negu_list = []
    for u in QPRu_list:
        negu_list.append(ne.negu(u))

    #transposed unitaries
    QPRulist_transposed = []
    for gate in range( len(QPRu_list) ):
        QPRulist_transposed.append(np.transpose(QPRu_list[gate]))

    start = time.time()
    samples = 0
    ptotal = 0
    while samples < samples_needed:
        
        ptotal += p(initial, QPRs_prob, QPRu_list, negu_list, QPRm, QPRulist_transposed)
        samples += 1
        
    end = time.time()
    print(end - start)
    #Approx 11.7s for 100k samples for 2 qutrit - Old
    #Approx 9.24s for 100k samples for 2 qutrit, 22.33s for 3 qutrit
    
    return [ptotal/samples, samples_needed]



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
