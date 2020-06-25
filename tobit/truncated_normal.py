from scipy.stats import norm
from util.data import normalize
from util.display import plot_gausian, plot_pdf
import matplotlib.pyplot as plt
from math import inf
import numpy as np

def pdf(input, mean = 0, std = 1, lower = 0, upper = None):
    x = normalize(input, mean, std)
    low = normalize(lower, mean, std)
    high = normalize(upper, mean, std)

    gate_low = 1 if low == None else x > low
    gate_high = 1 if high == None else x < high
    gate = gate_low * gate_high

    pdf_x = norm.pdf(x) * gate
    cdf_high = 1 if high == None else norm.cdf(high)
    cdf_low = 0 if low == None else norm.cdf(low)

    return pdf_x / (std * (cdf_high - cdf_low))

if __name__ == '__main__':
    mean, std = -10, 10
    plot_gausian(mean, std)
    plot_pdf(mean, std, lambda x: pdf(x, mean, std))
    plt.show()
