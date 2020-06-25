import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

low_plot_bound, high_plot_bound = 10, 10

def plot_gausian(mean, std):
    x = np.linspace(mean - low_plot_bound * std, mean + high_plot_bound * std, 1000)
    plt.plot(x, norm.pdf(x, mean, std))

def plot_pdf(mean, std, pdf):
    x = np.linspace(mean - low_plot_bound * std, mean + high_plot_bound * std, 1000)
    plt.plot(x, pdf(x))