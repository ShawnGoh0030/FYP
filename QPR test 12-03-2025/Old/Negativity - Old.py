import Wigner as wig
import QPR as qr
import numpy as np

#Negativity of state (absolute values of QPR of state summed)
def negs(QPRs):

    summ = 0
    for entry in QPRs:
        summ += abs(entry)
    return summ

#Negativity of unitary (sums of absolute values of a column in S matrix)
def negu(QPRu):

    length = len(QPRu[0])
    summs = []
    for i in range(length):
        column_summ = 0
        for j in range(length):
            column_summ += abs( QPRu[j][i] )
        summs.append(column_summ)
    return(summs)

#Negativity of measure (absolute value of QPR of measurement summed)
def negm(QPRm):

    summ = 0
    for entry in QPRm:
        summ += abs(entry)
    return summ

