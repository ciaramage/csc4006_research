from helpers.matrix_in_out import read_matrix
from texttable import Texttable
import latextable

def realDataInfo():
    header = ['Dataset', 'm', 'v']
    file_names = ['X50sites', 'Xpitprops', 'wdbc','qsar', 'adelaideWaves']
    data_names = ['Wave Sites', 'Pitprops', 'Breast Cancer', 'QSAR', 'Wave Energy Converters']
    
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
    table.set_deco(Texttable.HEADER | Texttable.HLINES)
    table.add_rows(rows)

    print('\nLatek Table: ')
    print(latextable.draw_latex(table, caption="Overview of the real data used in this study"))

def randomDataInfo():
    header = ['Dataset', 'm', 'v']
    rows = []
    rows.append(header)
    for i in range(11):
        mat = read_matrix('data/randomData/t{0}.txt'.format((i+1)))
        m, v = mat.shape
        dataset = 't{0}'.format((i+1))
        rows.append([dataset, m, v])
    print(rows)

    table = Texttable()
    table.set_cols_align(["c"] * 3)
    table.set_deco(Texttable.HEADER | Texttable.HLINES)
    table.add_rows(rows)

    print('\nLatek Table: ')
    print(latextable.draw_latex(table, caption="Overview of the random data used in this study"))
