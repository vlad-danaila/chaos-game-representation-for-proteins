'''
This is a diferentiable aproximation of the standard Gausian cumulative distribution function (cdf)
The aproximation is based on point 4 in:
http://www.hrpub.org/download/20140305/MS7-13401470.pdf
'''

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
    # cdf_aprox_plot()
    log_cdf_gradients_plot()