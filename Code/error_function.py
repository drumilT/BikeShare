
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
#import keras.backend as K

def punisher( pred, act):

    #as the function name this decides the ranges and tolerances that one is willing to accept as error


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

    if abs(act-pred) <= tol :
        error = 0
    else:
        error = (act-pred) * (act-pred)

    return error

def travel_data_accuracy( labelS, predS):

    # return the mean accuracy of an label and prediction matrix
    # accuracy is 1 if the value is well within the tolerance as defined in punisher, else 0

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
    # return the mean error of an label and prediction matrix
    # error is 0 if the value is well within the tolerance as defined in punisher, else its mse

    erlst = []
    for i in range(len(labelS)):
        erlst.append(punisher(predS[i], labelS[i]))

    return np.mean(np.array(erlst))


#### Everything defined below is primary made for the model to train on using only backend functions
# however is very heavy and slow due to reasons not known
# functions same as the above defined functions
@tf.function
def tol(act):
    ranges = [[0, 5], [6, 10], [11, 100]]
    tolerances = [2.0, 3.0, 5.0]
    comb = zip(ranges, tolerances)

    tol = 2.0

    for (low, up), tolerance in comb:
        if K.less(act, K.constant(up)):
            tol = tolerance

    #if isinstance(tol, int):
      #  tol = tol
    #else:
     #   tol = tol * act

    return tol

def mapper(val, key):
    return val + key*tol(val)

def mapperH(val):
    return mapper(val,1)

def mapperU(val):
    return mapper(val,-1)

@tf.function
def check_neg(val):
    zero = K.constant(0.0)
    if K.less(val,zero) :
        return 4.0
    else:
        return 1.0
def travel_error(labelS, predS):

    labelrs = K.flatten(labelS)
    preds = K.flatten(predS)
    high = K.map_fn(mapperH,labelrs)
    low = K.map_fn(mapperU,labelrs)

    upper = high - preds
    lower =  preds - low

    upperts = K.map_fn(check_neg,upper)
    lowers = K.map_fn(check_neg,lower)

    sc = tf.tensordot(  lowers,upperts , axes=0)

    #sc = tf.map_fn(check_neg, labelrs)
    #mse = K.square(labelrs-preds)

    #scMSE = tf.tensordot(sc, mse, axes=0 )

    return K.mean(sc)