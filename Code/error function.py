import numpy as np


def punisher(pred, act):
    ranges = [[0, 5], [6, 10], [11, 100]]
    tolerances = [2, 3, 0.2]
    comb = zip(ranges,tolerances)

    tol = next( tolerance for (low,up) , tolerance in comb
            if low<= act <= up)

    if isinstance(tol, int) :
        tol = tol
    else:
        tol = tol * act

    error = 1000

    if abs(act-pred) < tol :
        error = 0
    else:
        error = (act-pred)^2

def travel_data_accuracy(predS, labelS):

    erlst =[]

    for p,l in zip(predS,labelS):
        erlst.append(punisher(p,l))

    acclst = []

    for err in erlst:
        if err == 0:
            acclst.append(1)
        else:
            acclst.append(0)

    return acclst


def travel_data_error(predS, labelS):
    erlst = []

    for p, l in zip(predS, labelS):
        erlst.append(punisher(p, l))

    acclst = []

    return erlst