import numpy as np
import matplotlib.pyplot as plt


def quick_plot_and_save(x, y, where_):
    plt.plot(x, y)
    plt.savefig(f"{where_},.png")
