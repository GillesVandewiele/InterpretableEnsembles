from imblearn.under_sampling import RandomUnderSampler
from pylatex import Document, Section, Subsection, Tabular, MultiColumn

def write_preamble():
    return '\\documentclass{article}\n' \
            '\usepackage[margin=0.1in]{geometry}\n' \
            '\usepackage{graphicx}\n\n' \
            '\\begin{document}\n\n'

def write_measurements(measurements):
    # measurements = {'dataset': {'C4.5': {'acc': 0.56, 'bal_acc': 0.47, 'stddev': 0.15} } }
    # figures = {'dataset': path }
    datasets = measurements.keys()
    algorithms = measurements[datasets[0]].keys()
    metrics = measurements[datasets[0]][algorithms[0]].keys()
    latex = '\\begin{table}\n' \
            '\\centering\n' \
            '\\resizebox{\\textwidth}{!}{%\n' \
            '\\begin{tabular}{|'+'c|'*(len(algorithms)*len(metrics)+1)+'}\n'

    nr_cols = len(metrics)
    latex_line = '\\hline'
    for algorithm in algorithms:
        latex_line += ' & \multicolumn{'+str(nr_cols)+'}{|c|}{\\textbf{'+algorithm+'}}'
    latex_line += ' \\\\ \\hline \n'
    latex += latex_line

    latex_line = ''
    for metric in metrics:
        latex_line += ' & \\textbf{'+metric+'}'
    latex_line = latex_line * len(algorithms)
    latex_line += ' \\\\ \\hline \n'
    latex += latex_line

    for dataset in datasets:
        latex_line = '\\textbf{'+dataset+'} '
        for algorithm in algorithms:
            for metric in metrics:
                latex_line += ' & ' + str(measurements[dataset][algorithm][metric][0]) + '+' + \
                              str(measurements[dataset][algorithm][metric][1]) + '$\\sigma$'
        latex_line += ' \\\\ \\hline \n'
        latex += latex_line

    latex += '\\end{tabular}}\n' \
             '\\end{table}\n\n'
    return latex

def write_figures(figures):
    latex = ''
    for key in figures:
        latex += '\\begin{figure}\\centering\\includegraphics[width=\\textwidth]{'+figures[key]+'}\caption{Confusion matrix for '+key+'}\end{figure}\n'
    return latex

def write_footing():
    return '\\end{document}'
