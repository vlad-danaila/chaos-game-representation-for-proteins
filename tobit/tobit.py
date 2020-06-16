from data.assay_reader import Assay
import portion as p
import torch as t
from typing import List
import constants
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

def pdf(n):
    return ( 1 / math.sqrt(2 * math.pi) ) * t.exp( (-1/2) * (n ** 2) )

def negative_log_likelihood(y, mean, var):
    return -t.sum(t.log((1 / var) * (pdf((y - mean) / var)) + 1e-8))


def read_tensors_from_assay_intervals(intervals: List[p.interval.Interval]):
    single_valued = []
    for interval in intervals:
        # If single valued interval
        if interval.lower == interval.upper:
            single_valued.append(interval.lower)
    return t.tensor(single_valued, dtype=t.float, device=constants.DEVICE)

if __name__ == '__main__':
    assay = Assay('', '', [ p.singleton(100), p.singleton(150), p.singleton(150) ], None)
    tensors = read_tensors_from_assay_intervals(assay.ic50)
    mean, var = t.tensor(100, dtype=float, requires_grad=True), t.tensor(50, dtype=float, requires_grad=True)
    optimizer = t.optim.SGD([mean, var], lr = 1e+0)
    for i in range(10_000):
        optimizer.zero_grad()
        log_likelihood = negative_log_likelihood(tensors, mean, var)
        log_likelihood.backward()
        optimizer.step()
    print(mean, var)
    print('Expected', norm.fit(tensors))
