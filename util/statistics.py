import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from util.display import plot_gausian, plot_truncated_gausian
import matplotlib.pyplot as plt

def norm_pdf_intersect(x, mean1, std1, mean2, std2):
    pdf_1 = norm.pdf(x, mean1, std1)
    pdf_2 = norm.pdf(x, mean2, std2)
    return min(pdf_1, pdf_2)

def norm_pdf_union(x, mean1, std1, mean2, std2):
    pdf_1 = norm.pdf(x, mean1, std1)
    pdf_2 = norm.pdf(x, mean2, std2)
    return max(pdf_1, pdf_2)

def norm_iou(mean1, std1, mean2, std2):
    integral_norm_intersect, err1 = quad(norm_pdf_intersect, -np.inf, np.inf, args=(mean1, std1, mean2, std2))
    integral_norm_union, err2 = quad(norm_pdf_union, -np.inf, np.inf, args=(mean1, std1, mean2, std2))
    return integral_norm_intersect / integral_norm_union

if __name__ == '__main__':
    mean_1, std_1 = 10, 7
    # plot_gausian(mean_1, std_1)
    plot_truncated_gausian(mean_1, std_1)

    mean_2, std_2 = 10, 6
    # plot_gausian(mean_2, std_2)
    plot_truncated_gausian(mean_2, std_2)

    print(norm_iou(mean_1, std_1, mean_2, std_2))

    plt.show()
