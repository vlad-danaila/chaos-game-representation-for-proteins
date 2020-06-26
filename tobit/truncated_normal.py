from scipy.stats import norm
from util.data import normalize
import matplotlib.pyplot as plt
import numpy as np

def truncated_normal_pdf(input, mean = 0, std = 1, lower = 0, upper = None):
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

low_plot_bound, high_plot_bound = 10, 10

if __name__ == '__main__':
    mean, std = -10, 10
    x = np.linspace(mean - low_plot_bound * std, mean + high_plot_bound * std, 1000)
    plt.plot(x, norm.pdf(x, mean, std))
    plt.plot(x, truncated_normal_pdf(x, mean, std))
    plt.show()