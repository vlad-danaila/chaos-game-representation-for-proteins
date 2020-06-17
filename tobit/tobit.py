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

def tobit_mean_and_variance_reparametrization(intervals: List[p.interval.Interval]):
    single_val_tensors = read_tensors_from_assay_intervals(intervals)
    single_val_mean, single_val_std = single_val_tensors.mean(), single_val_tensors.std(unbiased = False)
    single_val_tensors = (single_val_tensors - single_val_mean) / single_val_std
    delta, gamma = t.tensor(0, dtype=float, requires_grad=True), t.tensor(1, dtype=float, requires_grad=True)
    optimizer = t.optim.SGD([delta, gamma], lr=1e-4)
    patience = 5
    for i in range(100_000):
        prev_delta, prev_gamma = delta.clone(), gamma.clone()
        optimizer.zero_grad()
        log_likelihood = negative_log_likelihood_reparametized(single_val_tensors, delta, gamma)
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

def gausian_curves_1():
    ic50 = [ p.singleton(30), p.singleton(30), p.singleton(50) ]
    mean, std = norm.fit(read_tensors_from_assay_intervals(ic50))
    x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
    plt.plot(x, norm.pdf(x, mean, std))
    # plt.plot(x, norm.cdf(x, mean, std))

    ic50 = [ p.singleton(30), p.singleton(30), p.singleton(60) ]
    mean, std = norm.fit(read_tensors_from_assay_intervals(ic50))
    x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
    plt.plot(x, norm.pdf(x, mean, std))
    # plt.plot(x, norm.cdf(x, mean, std))

    ic50 = [ p.singleton(30), p.singleton(30), p.singleton(70) ]
    mean, std = norm.fit(read_tensors_from_assay_intervals(ic50))
    x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
    plt.plot(x, norm.pdf(x, mean, std))
    # plt.plot(x, norm.cdf(x, mean, std))

    plt.show()

# http://www.hrpub.org/download/20140305/MS7-13401470.pdf
def cdf_aproximation_1(x):
    k = sqrt(2 / pi)
    return t.exp(2 * k * x) / (1 + t.exp(2 * k * x))

# http://www.hrpub.org/download/20140305/MS7-13401470.pdf
def cdf_aproximation_4(x: t.Tensor):
    y = sqrt(2/pi) * x * (1 + 0.044715 * x.pow(2))
    return .5 * (1 + tanh(y))

# http://www.hrpub.org/download/20140305/MS7-13401470.pdf
def cdf_aproximation_5(x: t.Tensor):
    y = 0.806 * x * (1 - 0.018 * x)
    return .5 * (1 - t.sqrt(1 - t.exp(-y.pow(2))))

def cdf_aprox_combined(x: t.Tensor):
    r = .015
    return r * cdf_aproximation_1(x) + (1 - r) * cdf_aproximation_4(x)

def tanh(x: t.Tensor):
    return (t.exp(x) - t.exp(-x)) / (t.exp(x) + t.exp(-x))

def cdf_aprox_plot():
    x = np.linspace(-4, 4, 1000)
    plt.plot(x, norm.cdf(x))
    plt.plot(x, cdf_aprox_combined(t.tensor(x, dtype=t.float64)))
    plt.show()

def log_cdf_gradients_plot():
    x = np.linspace(-6, 6, 1000)
    plt.plot(x, norm.pdf(x)/norm.cdf(x))
    plt.plot(x, compute_gradients_for_log_aprox_cdf(x))
    plt.show()

def compute_gradients_for_log_aprox_cdf(x):
    grads = []
    for _x in x:
        _x = t.tensor(_x, dtype=float, requires_grad=True)
        z = t.log(cdf_aproximation_4(_x))
        z.backward()
        grads.append(_x.grad)
    return grads

if __name__ == '__main__':
    # assay = Assay('', '', [ p.singleton(150), p.singleton(100), p.singleton(150) ], None)
    # print(tobit_mean_and_variance_reparametrization(assay.ic50))
    # print('Expected', norm.fit(read_tensors_from_assay_intervals(assay.ic50)))

    # gausian_curves_1()
    # cdf_aprox_plot()
    log_cdf_gradients_plot()

    # tensor = t.tensor([0, 100, 100, 100, 100, 100, 100], dtype=t.float)
    # single_val_mean, single_val_std = tensor.mean(), tensor.std(unbiased=False)
    # tensor = (tensor - single_val_mean) / single_val_std
    # print(tensor)
