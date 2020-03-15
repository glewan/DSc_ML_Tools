import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

CMAP = plt.cm.Blues

def multiple_line_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, legend_title: str, xlabel: str, ylabel: str, percentage=False):
    legend: list = []
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)

    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    # ax.legend(legend, title=legend_title, loc='center left', bbox_to_anchor=(1, 0.8))
    ax.legend(legend, title=legend_title, loc='upper right')


def multiple_bar_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str, percentage=False):

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    x = np.arange(len(xvalues))  # the label locations
    ax.set_xticks(x)
    ax.set_xticklabels(xvalues, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    width = 0.8  # the width of the bars
    step = width / len(yvalues)
    k = 0
    for name, y in yvalues.items():
        ax.bar(x + k * step, y, step, label=name)
        k += 1
    ax.legend(loc='lower center', ncol=len(yvalues), bbox_to_anchor=(0.5, -0.2), fancybox = True, shadow = True)

def plot_confusion_matrix(ax: plt.Axes, cnf_matrix: np.ndarray, legend_title: str, classes_names: list, normalize: bool = False):
    if normalize:
        total = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / total
        title = legend_title
    else:
        cm = cnf_matrix
        title = legend_title
    np.set_printoptions(precision=2)
    tick_marks = np.arange(0, len(classes_names), 1)
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes_names)
    ax.set_yticklabels(classes_names)
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    fmt = '.2f' # if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt), horizontalalignment="center")

def show_tree(trnX, trnY):
    from sklearn.tree import export_graphviz

    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(trnX, trnY)

    dot_data = export_graphviz(tree, out_file='dtree.dot', filled=True, rounded=True, special_characters=True)
    # Convert to png
    from subprocess import call
    call(['dot', '-Tpng', '../dtree.dot', '-o', './dtree.png', '-Gdpi=600'])

    plt.figure(figsize=(14, 18))
    plt.imshow(plt.imread('./dtree.png'))
    plt.axis('off')
    plt.show()

def line_chart(ax: plt.Axes, series: pd.Series, title: str, xlabel: str, ylabel: str, percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.plot(series)

