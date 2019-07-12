from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import keras.backend as K

def punisher( pred, act):

    #print(type(pred))
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
        error = (act-pred) * (act-pred)

    return error

def travel_data_accuracy( labelS, predS):

    #print(type(predS))

    erlst =[]

    for i in range(len(labelS)):
        erlst.append(punisher(predS[i], labelS[i]))

    acclst = []

    for err in erlst:
        if err == 0:
            acclst.append(1)
        else:
            acclst.append(0)

    return np.mean(np.array(acclst))

def travel_data_error( labelS, predS):

    erlst = []
    for i in range(len(labelS)):
        erlst.append(punisher(predS[i], labelS[i]))

    return np.mean(np.array(erlst))












def punisher2( arr):

    ranges = [[0, 5], [6, 10], [11, 100]]
    tolerances = [2, 3, 0.2]
    comb = zip(ranges,tolerances)

    pred = arr[1]
    act = arr[0]

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

def travel_error(labelS, predS):
    return K.mean(predS)
