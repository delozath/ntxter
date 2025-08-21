import seaborn as sns
from matplotlib import pyplot as plt

def pairplot(df, x_vars=None, y_vars=None):
    params = {
        'corner': True,
        'height': 1.5,
        'plot_kws': {"s": 10}
    }
    params |= {'x_vars': x_vars, 'y_vars': y_vars} if not((x_vars is None) or (y_vars is None)) else {}
    breakpoint()
    g = sns.pairplot(df, **params)
    #
    for ax in g.axes.flatten():
        if ax:
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
    #
    #for i, ax in enumerate(g.axes[-1]):
    #    if ax:
    #        ax.set_xlabel(ax.get_xlabel(),  rotation=45, fontsize=7, labelpad=10)
    #
    for i in range(len(g.axes)):
        ax = g.axes[i][0]
        if ax:
            ax.set_ylabel(ax.get_ylabel(), fontsize=7,  rotation=0, labelpad=10)
    #
    g.figure.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1)