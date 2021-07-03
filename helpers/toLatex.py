
import latextable
import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib
from helpers.matrix_in_out import read_matrix
from texttable import Texttable
from helpers.experiment_results import *

def realDataInfo():
    header = ['Dataset', 'm', 'v']
    file_names = ['X50sites', 'Xpitprops', 'wdbc', 'dryBeans', 'frogs']
    data_names = ['Wave Sites', 'Pitprops', 'Breast Cancer Diagnosis', 'Dry Beans', 'Anuran Frog Calls']
    
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
    golden_ratio = (5**.5 - 1.0)/2.0 # Golden ration to set aesthetic figure height
    fig_width = fig_width_pt * inches_per_pt # width in inches
    fig_height = fig_width * golden_ratio # height in inches
    fig_size = [fig_width,fig_height]
    
    # update params
    matplotlib.rcParams.update({
    'axes.labelsize': 10,
    'font.size': 10,
    'font.family':'serif',
    'legend.fontsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8})

    return fig_size

def plot_random_times():
    names = ['fsca', 'opfs', 'ufs']
    for k in names:
        fig_size = setupFigure()
        plt.figure(figsize=fig_size)

        # plt.title("Duration of {0} algorithms with different sizes of data matrices".format(k.upper()))
        plt.xlabel("Number of Columns")
        plt.ylabel("Duration(s)")

        x, duration = random_duration(k)

        for j in duration.keys():
            plt.plot(x, duration[j], label=j)

        plt.legend()

        tikzplotlib.save("output/random/randomTime{0}.tex".format(k.upper()))


def plot_real(alg_type, dataset):

    ds, duration, varEx, compID = real_results(alg_type, dataset)

    gLg_compID_table(ds, compID)

    """ for k in duration.keys():
        print(k)
        print(duration[k])
        print(compID[k])
        print(varEx[k])
        print('\n') """
    
def gLg_compID_table(ds, compID):
    header = ['k']
    for k in compID.keys():
        header.append(k)

    rows = []
    rows.append(header)

    for i in range(6):
        row = []
        row.append(i+1)
        for k in compID.keys():
            row.append(compID[k][i])
        rows.append(row)

    table = Texttable()
    table.set_cols_align(["c"] * len(rows[0]))
    table.set_deco(Texttable.HEADER)
    table.add_rows(rows)

    outputLatex = latextable.draw_latex(table, caption="Indices of variables selected by the greedy and lazy greedy implementations for the {0} dataset and for k = 1,..,6 the kth selected variable is indicated".format(ds))
    # save output to latex file in output/random/ds/glg{dataset}.tex
    with open('output/real/{0}/glgCompID.tex'.format(ds), 'w') as file:
        file.write(outputLatex)


#def sg_compID_table(ds, percent, compID):
    





