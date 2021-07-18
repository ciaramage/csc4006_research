
import latextable
import matplotlib
import matplotlib.pyplot as plt
from numpy import real
import tikzplotlib
from helpers.matrix_in_out import read_matrix
from texttable import Texttable
from helpers.experiment_results import *

def realDataInfo():
    """ This function creates a table containing the dimensions of each of the real datasets used in the research study and outputs to a Tex file. 
    """
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

    outputLatex = latextable.draw_latex(latextable.draw_latex(table, caption="Overview of the real data used in this study"))
    # save output to latex file in output/random/ds/glg{dataset}.tex
    with open('output/real/data_dimensions.tex','w') as file:
        file.write(outputLatex)

def randomDataInfo():
    """ This function creates a table containing the dimensions of each of the random datasets used in the research study and outputs to a Tex file."""
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

    outputLatex = latextable.draw_latex(latextable.draw_latex(table, caption="Overview of the random data used in this study"))
    # save output to latex file in output/random/ds/glg{dataset}.tex
    with open('output/random/data_dimensions.tex','w') as file:
        file.write(outputLatex)

def setupFigure():
    """ Returns the size of a figure to plot using matplotlib. 
    The width of the column is taken from the width of 1 column in the Research Article, and the golden aspect ratio is used to determine the height.

    Returns:
        [float, float]: Width, height in inches.
    """
    # Setup size of figure
    fig_width_pt = 252.0 # Get from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27 # Convert pt to inches
    golden_ratio = (5**.5 - 1.0)/2.0 # Golden ration to set aesthetic figure height
    fig_width = fig_width_pt * inches_per_pt # width in inches
    fig_height = fig_width * golden_ratio # height in inches
    fig_size = [fig_width,fig_height]
    
    # Update matplotlib rc params
    matplotlib.rcParams.update({
    'axes.labelsize': 10,
    'font.size': 10,
    'font.family':'serif',
    'legend.fontsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8})

    return fig_size

def plot_random_times():
    """ Plots the duration of performing the algorithms and their optimised implementations on the random datasets.
    Uses the tikzplotlib library to save the matplotlib plot to a Tek file that can be imported to the Research Article.
    """
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
    """ Takes output from performing algorithms 'alg_type' on datasets 'dataset' and call relevant functions that 
    will use the results to create tables and charts and output them to Tex files which can be imported to the Research Article.
    """

    #ds, duration, varEx, compID = real_results(alg_type, dataset, 0.4)

    #gLg_compID_table(ds, compID)
    #gLg_varEx_table(ds, varEx)
    #sg_compID_table(ds, compID)

    """ for k in duration.keys():
        print(k)
        print(duration[k])
        print(compID[k])
        print(varEx[k])
        print('\n') """

def plot_real_sg_compare(dataset, percentages):

    ds, duration, varEx, compID = real_sg_compare_results(dataset, percentages)

    #sg_compare_duration_table(ds, duration, percentages)
    #sg_compare_duration_graph(ds, duration, percentages)
    #sg_compare_compID_table(ds, compID, percentages)
    sg_compare_varEx_table(ds, varEx, percentages)


    
def gLg_compID_table(ds, compID):
    header = ['k']
    for k in compID.keys():
        header.append(k)

    rows = [header]

    for i in range(6):
        row = [i+1]
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

def gLg_varEx_table(ds, varEx):
    header = ['k']
    for k in varEx.keys():
        header.append(k)
    rows = [header]

    for i in range(6):
        row = [i+1]
        for k in varEx.keys():
            row.append(varEx[k][i])
        rows.append(row)
    
    table= Texttable()
    table.set_cols_align(["c"] * len(rows[0]))
    table.set_deco(Texttable.HEADER)
    table.add_rows(rows)

    outputLatek = latextable.draw_latex(table, caption="The variance explained by each selected component of the greedy and lazy greedy implementations for the {0} dataset and for k = 1,..,6 the kth selected variable is indicated".format(ds))
    with open('output/real/{0}/glgVarEx.tex'.format(ds), 'w') as file:
        file.write(outputLatek)

def sg_compID_table(ds, compID):
    header = ['K', 1]
    rows = [header]
    x = 2
    for k in compID.keys():
        row = []
        row.append(k)
        header.append(x)
        x = x + 1
        for i in range(len(compID[k])):
            row.append(compID[k][i])
        rows.append(row)

    table = Texttable()
    table.set_cols_align(["c"] * len(rows[0]))
    table.set_deco(Texttable.HEADER)
    table.add_rows(rows)

    outputLatex = latextable.draw_latex(table, caption="Indices of variables selected by the stochastic greedy implementations for the {0} dataset and for k = 1,..,6 the kth selected variable is indicated".format(ds))
    # save output to latex file in output/random/ds/glg{dataset}.tex
    with open('output/real/{0}/sgCompID.tex'.format(ds), 'w') as file:
        file.write(outputLatex)

def sg_compare_duration_table(ds, duration, percentages):
    # Table Header
    header =['Algorithms']
    for p in percentages:
        header.append(p)
    
    # Table Rows
    rows = [header]
    for k in duration.keys():
        row = [k]
        for p in percentages:
            row.append(duration[k][p])
        rows.append(row)

    # Construct Table
    table = Texttable()
    table.set_cols_align(["c"] * len(rows[0]))
    table.set_deco(Texttable.HEADER)
    table.add_rows(rows, header=True)

    # Create output
    outputLatex = latextable.draw_latex(table, caption="Computation time (in seconds) to perform Stochastic Greedy feature selection on the {0} dataset with different percentages used to sample the data".format(ds))
    # Write to file
    with open('output/real/{0}/sg_compareDurationTable.tex'.format(ds), 'w') as file:
        file.write(outputLatex)
    
def sg_compare_duration_graph(ds, duration, percentages):
    # Setup graph
    fig_size = setupFigure()
    plt.figure(figsize=fig_size)

    # Storage for plot data. < key, value> = <algorithm, duration with sample>
    rowsDict = defaultdict(list)
    for k in duration.keys():
        row = []
        for p in percentages:
            row.append(duration[k][p])
        rowsDict[k] = row
    
    # Plot the rows
    for k in rowsDict.keys():
        plt.plot(percentages, rowsDict[k], label=k)
    
    plt.xlabel('Sample Percentage')
    plt.ylabel('Computation Time (in seconds)')
    plt.legend()

    tikzplotlib.save("output/real/{0}/sg_compareDurationGraph.tex".format(ds))

def sg_compare_compID_table(ds, compID, percentages):
    # Table Header
    header = ['\%']
    for i in range(Nc):
        header.append(i+1)
    
    # Table Rows
    for k in compID.keys():
        rows = [header]
        for p in percentages:
            row = [p]
            res = compID[k][p]
            for x in res:
                row.append(x)
            rows.append(row)

        # Setup table
        table = Texttable()
        table.set_cols_align(["c"]* len(header))
        table.set_deco(Texttable.HEADER)
        table.add_rows(rows)

        # Create output
        outputLatex = latextable.draw_latex(table, caption="Indices of variables selected by the {0} algorithm for the {1} dataset, using different random sampling percentages, and for k = 1,..,6 the kth selected variable is indicated".format(k, ds))
        # Write to file
        with open('output/real/{0}/sg_compareCompID_{1}.tex'.format(ds, k), 'w') as file:
            file.write(outputLatex)

def sg_compare_varEx_table(ds, varEx, percentages):
    # Table Header
    header = ['\%']
    for i in range(Nc):
        header.append(i+1)
    
    # Table Rows
    for k in varEx.keys():
        rows = [header]
        for p in percentages:
            row = [p]
            res = varEx[k][p]
            for x in res:
                row.append(x)
            rows.append(row)

        # Setup table
        table = Texttable()
        table.set_cols_align(["c"]* len(header))
        table.set_deco(Texttable.HEADER)
        table.add_rows(rows)

        # Create output
        outputLatex = latextable.draw_latex(table, caption="Variance explained by the variables selected by the {0} algorithm for the {1} dataset, using different random sampling percentages, and for k = 1,..,6 the kth selected variable is indicated".format(k, ds))
        # Write to file
        with open('output/real/{0}/sg_compareVarEx_{1}.tex'.format(ds, k), 'w') as file:
            file.write(outputLatex)

def sg_sample_rows_table():
    header = ['']

    # rowsDick < key, value> = <dataset name, number of samples>
    rowsDict = defaultdict(int)
    for k in datasets.keys():
        X = read_matrix(datasets[k])
        rowsDict[k] = X.shape[0]
        header.append(k)

    # results for each row appended to rows
    rows = [header]

    for i in range(10, 110, 10):
        # results from each percentage becomes a row
        row = []
        row.append(i)
        for k in rowsDict.keys():
            row.append(int(rowsDict[k] * i * 0.01))
        rows.append(row)

    # Output tabulated results to Tex file
    table = Texttable()
    table.set_cols_align(["c"] * len(rows[0]))
    table.set_deco(Texttable.HEADER)
    table.add_rows(rows)

    outputLatex = latextable.draw_latex(table, caption="Number of rows when different percentages are used to select subsets of the datasets via random sampling with replacement")
    # save output to latex file in output/random/ds/glg{dataset}.tex
    with open('output/real/sg_Sizes.tex', 'w') as file:
        file.write(outputLatex)
    

def sg_sample_rows_graph():
    # graph
    fig_size = setupFigure()
    plt.figure(figsize=fig_size)

    # rowsDick < key, value> = <dataset name, number of samples>
    rowsDict = defaultdict(int)
    for k in datasets.keys():
        X = read_matrix(datasets[k])
        rowsDict[k] = X.shape[0]

    percents = np.arange(start=10, stop=110, step=10)
    for k in rowsDict.keys():
        sizes = []
        for p in percents:
            sizes.append(int(rowsDict[k] * p * 0.01))

        plt.plot(percents, sizes, label=k)
        #'Number of rows selected in random sampling of datasets with different percentages')

    plt.xlabel('Percentages')
    plt.ylabel('Number of rows in random sample')
    plt.legend()

    tikzplotlib.save("output/real/sg_sizesGraph.tex")

