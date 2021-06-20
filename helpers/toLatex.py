
import latextable
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt
from helpers.matrix_in_out import read_matrix
from texttable import Texttable

def realDataInfo():
    header = ['Dataset', 'm', 'v']
    file_names = ['X50sites', 'Xpitprops', 'wdbc','wpbc', 'dryBeans', 'forestFires']
    data_names = ['Wave Sites', 'Pitprops', 'Breast Cancer Diagnosis', 'Breast Cancer Prognosis', 'Dry Beans', 'Forest Fires']
    
    rows = []
    rows.append(header)
    for i in range(len(file_names)):
        mat = read_matrix('data/realData/{0}.txt'.format(file_names[i]))
        m, v = mat.shape
        dataset = data_names[i]
        rows.append([dataset, m, v])
    print(rows)

    table = Texttable()
    table.set_cols_align(["c"] * 3)
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)

    print('\nLatek Table: ')
    print(latextable.draw_latex(table, caption="Overview of the real data used in this study"))

def randomDataInfo():
    header = ['Dataset', 'm', 'v']
    rows = []
    rows.append(header)
    for i in range(10):
        mat = read_matrix('data/randomData/t{0}.txt'.format((i+1)))
        m, v = mat.shape
        dataset = 't{0}'.format((i+1))
        rows.append([dataset, m, v])
    print(rows)

    table = Texttable()
    table.set_cols_align(["c"] * 3)
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)

    print('\nLatek Table: ')
    print(latextable.draw_latex(table, caption="Overview of the random data used in this study"))

def setupFigure():
    # Setup size of figure
    fig_width_pt = 252.0 # Get form LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27 # Convert pt to inches
    golden_mean = (sqrt(5)-1.0)/2.0 # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt # width in inches
    fig_height = fig_width * golden_mean # height in inches
    fig_size = [fig_width,fig_height]
    
    # update params
    matplotlib.rc_params.update({'backend':'ps',
    'axes.labelsize': 10,
    'text.fontsize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': True,
    'figure.figsize': fig_size})

def plot_times(names, durations, sizes):
    setupFigure()
    plt.xlabel('Rows')
    plt.ylabel('Duration')
    plt.title('Duration of {0} algorithm with different sizes of data matrices'.format(names))
    plt.legend()
    plt.show()
    


