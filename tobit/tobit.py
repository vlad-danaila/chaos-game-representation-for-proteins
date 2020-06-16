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

def negative_log_likelihood(y, mean, std):
    return -t.sum(t.log((1 / std) * (pdf((y - mean) / std))))

def read_tensors_from_assay_intervals(intervals: List[p.interval.Interval]):
    single_valued = []
    for interval in intervals:
        # If single valued interval
        if interval.lower == interval.upper:
            single_valued.append(interval.lower)
    return t.tensor(single_valued, dtype=t.float, device=constants.DEVICE)

def tobit_mean_and_variance(intervals: List[p.interval.Interval]):
    single_val_tensors = read_tensors_from_assay_intervals(intervals)
    single_val_mean, single_val_std = single_val_tensors.mean(), single_val_tensors.std(unbiased = False)
    single_val_tensors = (single_val_tensors - single_val_mean) / single_val_std
    mean, std = t.tensor(0, dtype=float, requires_grad=True), t.tensor(1, dtype=float, requires_grad=True)
    optimizer = t.optim.SGD([mean, std], lr=1e-4)
    patience = 5
    for i in range(100_000):
        prev_mean, prev_std = mean.clone(), std.clone()
        optimizer.zero_grad()
        log_likelihood = negative_log_likelihood(single_val_tensors, mean, std)
        log_likelihood.backward()
        optimizer.step()
        early_stop = math.fabs(mean - prev_mean) + math.fabs(std - prev_std) < 1e-8
        if early_stop:
            patience -= 1
            if patience == 0:
                break
        else:
            patience = 5
    print(i)
    return mean + single_val_mean, std * single_val_std

if __name__ == '__main__':
    assay = Assay('', '', [ p.singleton(150), p.singleton(100), p.singleton(150) ], None)
    print(tobit_mean_and_variance(assay.ic50))
    print('Expected', norm.fit(read_tensors_from_assay_intervals(assay.ic50)))
