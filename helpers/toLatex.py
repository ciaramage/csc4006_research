
import latextable
import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib
from helpers.common import read_matrix_from_file
from texttable import Texttable
from helpers.experiment_results import *

####################################################
# Real Data Dimensions
####################################################
def realDataInfo():
    """ This function creates a table containing the dimensions of each of the real datasets used in the research study and outputs to a Tex file. 
    """
    header = ['Dataset', 'm', 'v']
    file_names = ['X50sites', 'Xpitprops', 'wdbc', 'frogs']
    data_names = ['Wave Sites', 'Pitprops', 'Breast Cancer Diagnosis', 'Anuran Frog Calls']
    
    rows = []
    rows.append(header)
    for i in range(len(file_names)):
        mat = read_matrix_from_file('data/realData/{0}.txt'.format(file_names[i]))
        m, v = mat.shape
        dataset = data_names[i]
        rows.append([dataset, m, v])

    table = Texttable()
    table.set_cols_align(["c"] * 3)
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)

    outputLatex = latextable.draw_latex(table, caption="Overview of the real data used in this study")
    # save output to latex file in output/random/ds/glg{dataset}.tex
    with open('output/real/data_dimensions.tex','w') as file:
        file.write(outputLatex)

    with open('output/notLatex/real/data_dimensions.txt','w') as file:
        file.write(table.draw())

####################################################
# Random Data Dimensions
####################################################
def randomDataInfo():
    """ This function creates a table containing the dimensions of each of the random datasets used in the research study and outputs to a Tex file."""
    header = ['Dataset', 'm', 'v']
    rows = []
    rows.append(header)
    for i in range(10):
        mat = read_matrix_from_file('data/randomData/t{0}.txt'.format((i+1)))
        m, v = mat.shape
        dataset = 't{0}'.format((i+1))
        rows.append([dataset, m, v])

    table = Texttable()
    table.set_cols_align(["c"] * 3)
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)

    outputLatex = latextable.draw_latex(table, caption="Overview of the random data used in this study")
    # save output to latex file in output/random/ds/glg{dataset}.tex
    with open('output/random/data_dimensions.tex','w') as file:
        file.write(outputLatex)

    with open('output/notLatex/random/data_dimensions.txt','w') as file:
        file.write(table.draw())
    

####################################################
# Settings for plots
####################################################
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

####################################################
# Random durations table
####################################################
def plot_random_times():
    """ Plots the duration of performing the algorithms and their optimised implementations on the random datasets.
    Uses the tikzplotlib library to save the matplotlib plot to a Tek file that can be imported to the Research Article.
    """
    names = ['fsca', 'opfs', 'ufs']
    for k in names:
        fig_size = setupFigure()
        plt.figure(figsize=fig_size)

        plt.xlabel("Number of Columns")
        plt.ylabel("Duration(s)")

        x, duration = random_duration(k)

        for j in duration.keys():
            plt.plot(x, duration[j], label=j)

        plt.legend()

        tikzplotlib.save("output/random/randomTime{0}.tex".format(k.upper()))

        plt.savefig("output/notLatex/random/randomTime{0}.png".format(k.upper()), format='png')

####################################################
# Greedy and lazy greedy with real datasets
####################################################
def plot_glg_real(alg_type, dataset):
    """ Takes output from performing algorithms 'alg_type' on datasets 'dataset' and call relevant functions that will use the results to create 
    tables and charts and output them to Tex files which can be imported to the Research Article.

    Args:
        alg_type (String): Algorithm to perform feature selectio with.
        dataset (String): Dataset to perform feature selection on.
        percentage (Float): Percentage value to be used in random sampling if/when stochastic optimisation is being applied.
    """
    ds, duration, varEx, compID = real_results(alg_type, dataset)

    gLg_compID_table(ds, compID)
    gLg_varEx_table(ds, varEx)
    gLg_duration_table(ds, duration)

    """ for k in duration.keys():
        print(k)
        print(duration[k])
        print(compID[k])
        print(varEx[k])
        print('\n') """

####################################################
# Stochastic greedy with real datasets and given 
# percentage (50%) for random sampling
####################################################
def plot_sg_real(alg_type, dataset, percentage):
    """ Takes output from performing algorithms 'alg_type' on datasets 'dataset' and call relevant functions that will use the results to create 
    tables and charts and output them to Tex files which can be imported to the Research Article.

    Args:
        alg_type (String): Algorithm to perform feature selectio with.
        dataset (String): Dataset to perform feature selection on.
        percentage (Float): Percentage value to be used in random sampling if/when stochastic optimisation is being applied.
    """
    ds, duration, varEx, compID = real_results(alg_type, dataset, percentage)
    
    sg_compID_table(ds, compID)
    sg_varEx_table(ds, varEx)
    sg_duration_table(ds, duration)

####################################################
# Stochastic greedy with real dataset and comparison
# of a range of percentages in random sampling
####################################################
def plot_real_sg_compare(dataset, percentages):
    """ Takes output from performing stochastic algorithms on dataset 'dataset' and call relevant functions that will use the results to create 
    tables and charts and output them to Tex files which can be imported to the Research Article.

    Args:
        dataset (String): Dataset to perform feature selection on.
        percentages (List of floats): Percentage values to be use random sampling.
    """
    ds, duration, varEx, compID = real_sg_compare_results(dataset, percentages)

    sg_compare_duration_table(ds, duration, percentages)
    sg_compare_duration_graph(ds, duration, percentages)
    sg_compare_compID_table(ds, compID, percentages)
    sg_compare_varEx_table(ds, varEx, percentages)

####################################################
# Table: Greedy + lazy greedy component ID
####################################################   
def gLg_compID_table(ds, compID):
    """ Accepts results containing selected component data and the name of the dataset the greedy and lazy algorithms selected the features from.
    Tabulates results and outputs to a Tex file which can be imported to the Research Article.

    Args:
        ds (String): Dataset feature selection was performed on.
        compID (Dictionary): Key - Algorithm type, Value - Component indexes of the features selected by that algorithm
    """
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

    outputLatex = latextable.draw_latex(table, caption="Indices of variables selected by the greedy and lazy greedy implementations for the {0} dataset and for k = 1,..,6 the kth selected variable is indicated".format(ds))
    # save output to latex file in output/random/ds/glg{dataset}.tex
    with open('output/real/{0}/glgCompID.tex'.format(ds), 'w') as file:
        file.write(outputLatex)

    with open('output/notLatex/real/{0}/glgCompID.txt'.format(ds),'w') as file:
        file.write(table.draw())

####################################################
# Table: Greedy + lazy greedy variance explained
####################################################   
def gLg_varEx_table(ds, varEx):
    """ Accepts results containing variance explained by selected component data and the name of the dataset the greedy and lazy greedy algorithms selected the features from.
    Tabulates results and outputs to a Tex file which can be imported to the Research Article.

    Args:
        ds (String): Dataset feature selection was performed on.
        varEx (Dictionary): Key - Algorithm type, Value - Variance explained by the features selected by that algorithm
    """
    header = ['K', 1]
    rows = [header]
    x = 2
    for k in varEx.keys():
        row = []
        row.append(k)
        header.append(x)
        x = x + 1
        for i in range(len(varEx[k])):
            row.append(varEx[k][i])
        rows.append(row)

    table = Texttable()
    table.set_cols_align(["c"] * len(rows[0]))
    table.set_deco(Texttable.HEADER)
    table.add_rows(rows)

    outputLatek = latextable.draw_latex(table, caption="The variance explained by each selected component of the greedy and lazy greedy implementations for the {0} dataset and for k = 1,..,6 the kth selected variable is indicated".format(ds))
    with open('output/real/{0}/glgVarEx.tex'.format(ds), 'w') as file:
        file.write(outputLatek)

    with open('output/notLatex/real/{0}/glgVarEx.txt'.format(ds),'w') as file:
        file.write(table.draw())


####################################################
# Table: Greedy + lazy greedy duration
####################################################   
def gLg_duration_table(ds, duration):
    header = ['Algorithm', 'Duration']
    rows = [header]
    for k in duration.keys():
        row = []
        row.append(k)
        row.append(str(duration[k][0]))
        rows.append(row)

    table = Texttable()
    table.set_cols_dtype(['t','t'])
    table.set_cols_align(["c"] * len(rows[0]))
    table.set_deco(Texttable.HEADER)
    table.add_rows(rows)

    outputLatex = latextable.draw_latex(table, caption="Computational time in seconds to select 6 features with the unsupervised selection algorithms".format(ds))
    # save output to latex file in output/random/ds/glg{dataset}.tex
    with open('output/real/{0}/glgDuration.tex'.format(ds), 'w') as file:
        file.write(outputLatex)

    with open('output/notLatex/real/{0}/glgDuration.txt'.format(ds),'w') as file:
        file.write(table.draw())

####################################################
# Table: Stochastic greedy component ID
####################################################   
def sg_compID_table(ds, compID):
    """ Accepts results containing selected component index data and the name of the dataset the stochastic greedy algorithms selected the features from.
    Tabulates results and outputs to a Tex file which can be imported to the Research Article.

    Args:
        ds (String): Dataset feature selection was performed on.
        compID (Dictionary): Key - Algorithm type, Value - Component indexes of the features selected by that algorithm
    """
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

    outputLatex = latextable.draw_latex(table, caption="Indices of variables selected by the stochastic greedy implementations for the {0} dataset and for k = 1,..,6 the kth selected variable is indicated using the default percentage for random sampling".format(ds))
    # save output to latex file in output/random/ds/glg{dataset}.tex
    with open('output/real/{0}/sgCompID.tex'.format(ds), 'w') as file:
        file.write(outputLatex)

    with open('output/notLatex/real/{0}/sgCompID.txt'.format(ds),'w') as file:
        file.write(table.draw())

####################################################
# Table: Stochastic greedy variance explained 
####################################################
def sg_varEx_table(ds, varEx):
    """ Accepts results containing selected component index data and the name of the dataset the stochastic greedy algorithms selected the features from.
    Tabulates results and outputs to a Tex file which can be imported to the Research Article.

    Args:
        ds (String): Dataset feature selection was performed on.
        varEx (Dictionary): Key - Algorithm type, Value - Component indexes of the features selected by that algorithm
    """
    header = ['K', 1]
    rows = [header]
    x = 2
    for k in varEx.keys():
        row = []
        row.append(k)
        header.append(x)
        x = x + 1
        for i in range(len(varEx[k])):
            row.append(varEx[k][i])
        rows.append(row)

    table = Texttable()
    table.set_cols_align(["c"] * len(rows[0]))
    table.set_deco(Texttable.HEADER)
    table.add_rows(rows)

    outputLatex = latextable.draw_latex(table, caption="Variance explained by variables selected by the stochastic greedy implementations for the {0} dataset and for k = 1,..,6 the kth selected variable is indicated using the default percentage for random sampling".format(ds))
    # save output to latex file in output/random/ds/glg{dataset}.tex
    with open('output/real/{0}/sgVarEx.tex'.format(ds), 'w') as file:
        file.write(outputLatex)
    
    with open('output/notLatex/real/{0}/sgVarEx.txt'.format(ds),'w') as file:
        file.write(table.draw())

####################################################
# Table: Stochastic greedy duration
####################################################
def sg_duration_table(ds, duration):
    header = ['Algorithm', 'Duration']
    rows = [header]
    for k in duration.keys():
        row = []
        row.append(k)
        row.append(str(duration[k][0]))
        rows.append(row)

    table = Texttable()
    table.set_cols_dtype(['t','t'])
    table.set_cols_align(["c"] * len(rows[0]))
    table.set_deco(Texttable.HEADER)
    table.add_rows(rows)

    outputLatex = latextable.draw_latex(table, caption="Computational time in seconds to select 6 features from the {0} dataset with the greedy and lazy greedy unsupervised stochastic selection algorithms".format(ds))
    # save output to latex file in output/random/ds/glg{dataset}.tex
    with open('output/real/{0}/sgDurationTable.tex'.format(ds), 'w') as file:
        file.write(outputLatex)

    with open('output/notLatex/real/{0}/sgDurationTable.txt'.format(ds),'w') as file:
        file.write(table.draw())

####################################################
# Table: Stochastic greedy percentage comparison
# duration
####################################################
def sg_compare_duration_table(ds, duration, percentages):
    """ Accepts results containing computation time data and the name of the dataset the stochastic greedy algorithms selected the features from.
    Tabulates results and outputs to a Tex file which can be imported to the Research Article.

    Args:
        ds (String): Dataset feature selection was performed on.
        duration (Dictionary): Key - algorithm type, Value - Dictionary: Key - percentage, Value - The computational time to perform the algorithm given in a list with percentage used in random sampling
        percentages (List of floats): Percentages values used in random sampling.
    """
    # Table Header
    header =['Algorithms']
    for p in percentages:
        header.append(p)
    
    # Table Rows
    rows = [header]
    for k in duration.keys():
        row = [k]
        for p in percentages:
            row.append(str(duration[k][p]))
        rows.append(row)

    # Construct Table
    table = Texttable()
    table.set_cols_dtype(['t','t','t'])
    table.set_cols_align(["c"] * len(rows[0]))
    table.set_deco(Texttable.HEADER)
    table.add_rows(rows, header=True)

    # Create output
    outputLatex = latextable.draw_latex(table, caption="Computation time (in seconds) to perform Stochastic Greedy feature selection on the {0} dataset with different percentages used to sample the data".format(ds))
    # Write to file
    with open('output/real/{0}/sg_compareDurationTable.tex'.format(ds), 'w') as file:
        file.write(outputLatex)

    with open('output/notLatex/real/{0}/sg_compareDurationTable.txt'.format(ds),'w') as file:
        file.write(table.draw())

####################################################
# Graph: Stochastic greedy percentage comparison
# duration
#################################################### 
def sg_compare_duration_graph(ds, duration, percentages):
    """ Accepts results containing computation time data and the name of the dataset the stochastic greedy algorithms selected the features from.
    Plots results and outputs to a Tex file which can be imported to the Research Article.

    Args:
        ds (String): Dataset feature selection was performed on.
        duration (Dictionary): Key - algorithm type, Value - Dictionary: Key - percentage, Value - The computational time to perform the algorithm given in a list with percentage used in random sampling
        percentages (List of floats): Percentages values used in random sampling.
    """
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

    plt.savefig("output/notLatex/real/{0}/sg_compareDurationGraph.png".format(ds), format='png')


####################################################
# Table: Stochastic greedy percentage comparison
# component ID
####################################################
def sg_compare_compID_table(ds, compID, percentages):
    """ Accepts results containing selected component index data and the name of the dataset the stochastic greedy algorithms selected the features from.
    Tabulates results and outputs to a Tex file which can be imported to the Research Article.

    Args:
        ds (String): Dataset feature selection was performed on.
        compID (Dictionary): Key - algorithm type, Value - Dictionary: Key - percentage, Value - The component indexes selected by the algorithms given in a list with percentage used in random sampling
        percentages (List of floats): Percentages values used in random sampling.
    """
    # Table Header
    header = ['\%']
    for i in range(Nc):
        header.append(i+1)
    
    # Table Rows
    for k in compID.keys():
        rows = [header]
        for p in percentages:
            row = [int(p*100)]
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
        
        with open('output/notLatex/real/{0}/sg_compareCompID.txt'.format(ds),'w') as file:
            file.write(table.draw())

####################################################
# Table: Stochastic greedy percentage comparison
# variance explained
####################################################
def sg_compare_varEx_table(ds, varEx, percentages):
    """ Accepts results containing variance explained by selected component data and the name of the dataset the stochastic greedy algorithms selected the features from.
    Tabulates results and outputs to a Tex file which can be imported to the Research Article.

    Args:
        ds (String): Dataset feature selection was performed on.
        varEx (Dictionary): Key - algorithm type, Value - Dictionary: Key - percentage, Value - The variance explained by components selected by the algorithms given in a list with percentage used in random sampling
        percentages (List of floats): Percentages values used in random sampling.
    """
    # Table Header
    header = ['\%']
    for i in range(Nc):
        header.append(i+1)
    
    # Table Rows
    for k in varEx.keys():
        rows = [header]
        for p in percentages:
            row = [int(p*100)]
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

        with open('output/notLatex/real/{0}/sg_compareVarEx.txt'.format(ds),'w') as file:
            file.write(table.draw())
        

####################################################
# Table: Number of rows in samples with percentage 
# of data to randomly sample incrementing by 10%
####################################################
def sg_sample_rows_table():
    """ Tabulates the number of rows in each random sample by using different percentages in random sampling.
    Outputs table to a Tex file which can be imported to the Research Article.
    """
    header = ['']

    # rowsDick < key, value> = <dataset name, number of samples>
    rowsDict = defaultdict(int)
    for k in datasets.keys():
        X = read_matrix_from_file(datasets[k])
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

    with open('output/notLatex/real/sg_Sizes.txt','w') as file:
        file.write(table.draw())
    
####################################################
# Graph: Number of rows in samples with percentage 
# of data to randomly sample incrementing by 10%
####################################################
def sg_sample_rows_graph():
    """ Plots the number of rows in each random sample by using different percentages in random sampling.
    Outputs plot to a Tex file which can be imported to the Research Article.
    """
    # graph
    fig_size = setupFigure()
    plt.figure(figsize=fig_size)

    # rowsDick < key, value> = <dataset name, number of samples>
    rowsDict = defaultdict(int)
    for k in datasets.keys():
        X = read_matrix_from_file(datasets[k])
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

    plt.savefig("output/notLatex/real/sg_sizesGraph.png", format='png')
