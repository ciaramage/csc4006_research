import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from pca_analysis.helpers.GraphTypes import GraphTypes

def generate_graph( g_type, g_data, g_name):
    if g_type is GraphTypes.SCATTER:
        gen_scatter(g_data, g_name)
    elif g_type is GraphTypes.SCREE:
        gen_scree(g_data, g_name)


def gen_scatter(g_data, g_name):
    plt.plot()

def gen_scree(g_data, g_name):
    plt.plot()
