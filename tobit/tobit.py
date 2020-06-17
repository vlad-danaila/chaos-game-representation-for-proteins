from data.assay_reader import Assay
import portion as p
import torch as t
from typing import List
import constants
import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi

def pdf(n):
    return ( 1 / math.sqrt(2 * math.pi) ) * t.exp( (-1/2) * (n ** 2) )

def negative_log_likelihood(y, mean, std):
    return -t.sum(t.log((1 / std) * (pdf((y - mean) / std))))

def negative_log_likelihood_reparametized(y, delta, gamma):
    return -t.sum(t.log(gamma) + t.log(pdf(gamma * y - delta)))

def read_tensors_from_assay_intervals(intervals: List[p.interval.Interval]):
    single_valued, right_censored, left_censored = [], [], []
    for interval in intervals:
        # If single valued interval
        if interval.lower == interval.upper:
            single_valued.append(interval.lower)
        elif interval.upper == p.inf:
            right_censored.append(interval.lower)
        elif interval.lower == 0:
            left_censored.append(interval.upper)
        else:
            raise Exception('Uncensored interval encountered', interval)
    return t.tensor(single_valued, dtype=t.float, device=constants.DEVICE), \
           t.tensor(right_censored, dtype=t.float, device=constants.DEVICE), \
           t.tensor(left_censored, dtype=t.float, device=constants.DEVICE)

def tobit_mean_and_variance(intervals: List[p.interval.Interval]):
    single_val, right_censored, left_censored = read_tensors_from_assay_intervals(intervals)
    single_val_mean, single_val_std = single_val.mean(), single_val.std(unbiased = False)
    single_val = (single_val - single_val_mean) / single_val_std
    mean, std = t.tensor(0, dtype=float, requires_grad=True), t.tensor(1, dtype=float, requires_grad=True)
    optimizer = t.optim.SGD([mean, std], lr=1e-4)
    patience = 5
    for i in range(100_000):
        prev_mean, prev_std = mean.clone(), std.clone()
        optimizer.zero_grad()
        log_likelihood = negative_log_likelihood(single_val, mean, std)
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

def tobit_mean_and_variance_reparametrization(intervals: List[p.interval.Interval]):
    single_val, right_censored, left_censored = read_tensors_from_assay_intervals(intervals)
    print(single_val, right_censored, left_censored)
    single_val_mean, single_val_std = single_val.mean(), single_val.std(unbiased = False)
    single_val = (single_val - single_val_mean) / single_val_std
    delta, gamma = t.tensor(0, dtype=float, requires_grad=True), t.tensor(1, dtype=float, requires_grad=True)
    optimizer = t.optim.SGD([delta, gamma], lr=1e-4)
    patience = 5
    for i in range(100_000):
        prev_delta, prev_gamma = delta.clone(), gamma.clone()
        optimizer.zero_grad()
        log_likelihood = negative_log_likelihood_reparametized(single_val, delta, gamma)
        log_likelihood.backward()
        optimizer.step()
        early_stop = math.fabs(delta - prev_delta) + math.fabs(gamma - prev_gamma) < 1e-8
        if early_stop:
            patience -= 1
            if patience == 0:
                break
        else:
            patience = 5
    mean, std = delta / gamma, gamma ** -2
    return mean + single_val_mean, std * single_val_std

def plot_gausian(mean, std):
    x = np.linspace(mean - 3 * std, mean + 3 * std, 1000)
    plt.plot(x, norm.pdf(x, mean, std))

def gausian_curves_1():
    for outlier in [50, 60, 70]:
        ic50 = [p.singleton(30), p.singleton(30), p.singleton(outlier)]
        mean, std = norm.fit(read_tensors_from_assay_intervals(ic50))
        x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
        plt.plot(x, norm.pdf(x, mean, std))
        # plt.plot(x, norm.cdf(x, mean, std))
    plt.show()

def to_numpy(tensor: t.Tensor):
    return tensor.clone().detach().numpy()

if __name__ == '__main__':
    assay = Assay('', '', [ p.singleton(10), p.singleton(20), p.singleton(30), p.closed(50, p.inf), p.closed(50, p.inf) ], None)
    mean, std = tobit_mean_and_variance_reparametrization(assay.ic50)
    print(mean, std)
    print('Expected', norm.fit(read_tensors_from_assay_intervals(assay.ic50)[0]))
    plot_gausian(to_numpy(mean), to_numpy(std))
    plt.show()