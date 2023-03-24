import numpy as np
from collections import namedtuple

DesiredState = namedtuple('DesiredState', 'pos vel acc jerk snap')

def polyder(t, k = 0, order = 10):
    if k == 'all':
        terms = np.array([polyder(t, k, order) for k in range(1,5)])
    else:
        terms = np.zeros(order)
        coeffs = np.polyder([1]*order, k)[::-1]
        pows = t**np.arange(0, order-k, 1)
        terms[k:] = coeffs*pows
    return terms

def Hessian(T, order = 10, opt = 4):
    n = len(T)
    Q = np.zeros((order*n, order*n))
    for k in range(n):
        m = np.arange(0, opt, 1)
        for i in range(order):
            for j in range(order):
                if i >= opt and j >= opt:
                    pow = i+j-2*opt+1
                    Q[order*k+i, order*k+j] = 2*np.prod((i-m)*(j-m))*T[k]**pow/pow
    return Q