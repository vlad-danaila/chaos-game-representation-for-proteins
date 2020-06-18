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
import tobit
from tobit.cdf_proximation import cdf_aproximation_4
from util.data import normalize

def pdf(n):
    return ( 1 / math.sqrt(2 * math.pi) ) * t.exp( (-1/2) * (n ** 2) )

def pdf_negative_log_likelihood_reparametized(y, delta, gamma):
    return -t.sum(t.log(gamma) + t.log(pdf(gamma * y - delta)))

def right_censored_cdf_negative_log_likelihood_reparametized(y, delta, gamma):
    return -t.sum(t.log(1 - cdf_aproximation_4(gamma * y - delta)))

# def zero_bound_cdf_negative_log_likelihood_reparametized(N, delta, gamma):
#     return -N * t.log(cdf_aproximation_4(gamma * y - delta))

def read_normalized_tensors_from_assay_intervals(intervals: List[p.interval.Interval]):
    single_valued, right_censored, left_censored, all = [], [], [], []

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

    all = np.array(single_valued + right_censored + left_censored)
    mean, std =  all.mean(), all.std() + 1e-10

    single_valued = t.tensor(single_valued, dtype=t.float, device=constants.DEVICE)
    right_censored = t.tensor(right_censored, dtype=t.float, device=constants.DEVICE)
    left_censored = t.tensor(left_censored, dtype=t.float, device=constants.DEVICE)

    single_valued = normalize(single_valued, mean, std)
    right_censored = normalize(right_censored, mean, std)
    left_censored = normalize(left_censored, mean, std)

    return single_valued, right_censored, left_censored, mean, std

def tobit_mean_and_variance_reparametrization(intervals: List[p.interval.Interval]):
    single_val, right_censored, left_censored, data_mean, data_std = read_normalized_tensors_from_assay_intervals(intervals)
    delta, gamma = t.tensor(0, dtype=float, requires_grad=True), t.tensor(1, dtype=float, requires_grad=True)
    optimizer = t.optim.SGD([delta, gamma], lr=1e-3)
    patience = 5
    for i in range(30_000):
        if i == 34461:
            print('here')
        prev_delta, prev_gamma = delta.clone(), gamma.clone()
        optimizer.zero_grad()
        log_likelihood = pdf_negative_log_likelihood_reparametized(single_val, delta, gamma) \
                         + right_censored_cdf_negative_log_likelihood_reparametized(right_censored, delta, gamma)
        log_likelihood.backward()
        optimizer.step()
        early_stop = math.fabs(delta - prev_delta) + math.fabs(gamma - prev_gamma) < 1e-5
        if early_stop:
            patience -= 1
            if patience == 0:
                break
        else:
            patience = 5
        print(i, delta, gamma)
    mean, std = delta / gamma, 1/gamma
    return mean + data_mean, std * data_std

def plot_gausian(mean, std):
    x = np.linspace(mean - 3 * std, mean + 3 * std, 1000)
    plt.plot(x, norm.pdf(x, mean, std))

def gausian_curves_1():
    for outlier in [50, 60, 70]:
        ic50 = [p.singleton(30), p.singleton(30), p.singleton(outlier)]
        mean, std = norm.fit(read_normalized_tensors_from_assay_intervals(ic50))
        x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
        plt.plot(x, norm.pdf(x, mean, std))
        # plt.plot(x, norm.cdf(x, mean, std))
    plt.show()

def to_numpy(tensor: t.Tensor):
    return tensor.clone().detach().numpy()

if __name__ == '__main__':
    no_tobit = np.array([30, 50, 50])
    no_tobit_mean, no_tobit_std = norm.fit(no_tobit)

    ic50 = [ p.singleton(30), p.closed(50, p.inf), p.closed(50, p.inf)]
    mean, std = tobit_mean_and_variance_reparametrization(ic50)

    print('No tobit mean', no_tobit_mean, 'std', no_tobit_std)
    plot_gausian(no_tobit_mean, no_tobit_std)
    print(mean, std)
    plot_gausian(to_numpy(mean), to_numpy(std))

    plt.show()

    # TODO : Bound variance at zero
    # TODO: Study the case of 50, (30, inf), (30, inf)
    # TODO: Handle left censoring

    # 30, 50, 50
    # 2410
    # 43.333333333333336 9.428090415820632
    # 44.8905 23.6967

    # 30, 50, 50, 50, 50
    # 2066
    # 46.0 8.0
    # 49.4213 29.4675