import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tobit.truncated_normal import truncated_normal_pdf

low_plot_bound, high_plot_bound = 10, 10

def plot_gausian(mean, std):
    x = np.linspace(mean - low_plot_bound * std, mean + high_plot_bound * std, 1000)
    plt.plot(x, norm.pdf(x, mean, std))

def plot_pdf(mean, std, pdf):
    x = np.linspace(mean - low_plot_bound * std, mean + high_plot_bound * std, 1000)
    plt.plot(x, pdf(x))

def plot_truncated_gausian(mean, std):
    plot_pdf(mean, std, lambda x: truncated_normal_pdf(x, mean, std))