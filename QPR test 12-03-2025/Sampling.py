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

#New sampling process
def new_sampling(QPRu_list, initial, QPRm):

    #initial QPRs
    entries = []
    probabilities = []
    length = len(initial)
    total_neg = ne.negs(initial)
    for i in range(length):
        if initial[i] == 0:
            pass
        else:
            entries.append((initial[i], i, initial[i] / total_neg))

    #each QPRu
    curr_terms = entries
    for QPRu in QPRu_list:
        curr_negu = ne.negu(QPRu)  #slowest step
        total_neg *= max(curr_negu)
        QPRu_transposed = np.transpose(QPRu)
        next_terms = []
        for term in curr_terms:
            #print(term)
            col_neg = curr_negu[term[1]]
            for i in range(length):
                temp = QPRu_transposed[term[1]][i]  #value of term
                if temp == 0:
                    pass
                else:
                    next_terms.append( (temp * term[0], i, term[2]*(temp/col_neg)) )
        curr_terms = next_terms

    #final QPRm and sampling
    absm = []
    for entry in QPRm:          #list of absolute values of m
        absm.append(abs(entry))
    total_neg *= max(absm)
    '''
    for entry in curr_terms:
        index = entry[1]
        if QPRm[entry[1]] == 0:
            pass
        else:
    '''        
    samples_needed = math.ceil(2 * (0.01**-2) * (total_neg**2) * np.log(2/0.05))
    probabilities = []
    values = []
    for entry in curr_terms:
        probabilities.append( abs(entry[2]) )
        values.append(entry[0] * QPRm[entry[1]])
    samples = np.random.choice(values, size = samples_needed, p = probabilities)
    p = np.sum(samples)
    return (p/samples_needed, samples_needed)
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
