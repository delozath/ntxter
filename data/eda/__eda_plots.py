import seaborn as sns
from matplotlib import pyplot as plt

def pairplot(df, **params):
    if not params:
        params = {
            'corner': True,
            'height': 1.5,
            'plot_kws': {"s": 10}
        }
    else:
        params |= {
            'height': 1.5,
            'plot_kws': {"s": 10}
        }
    #
    g = sns.pairplot(df, **params)
    #
    for ax in g.axes.flatten():
        if ax:
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
    #
    for i, ax in enumerate(g.axes[-1]):
        if ax:
            ax.set_xlabel(ax.get_xlabel(),  rotation=45, fontsize=7, labelpad=10)
    #
    for i in range(len(g.axes)):
        ax = g.axes[i][0]
        if ax:
            ax.set_ylabel(ax.get_ylabel(), fontsize=7,  rotation=0, labelpad=10)
    #
    g.figure.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1)
    g.figure.tight_layout(rect=[0, 0, 1, 1])

    