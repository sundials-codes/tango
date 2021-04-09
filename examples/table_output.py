#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=4)
    ax.tick_params(which="minor", bottom=False, left=False)

    # return im, cbar
    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def find_first(lst, a):

    for i, x in enumerate(lst):
        if x == a:
            return i


# ****** Main ***** #
def main():

    import os
    import argparse

    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # import statistics as stat

    parser = argparse.ArgumentParser(description='Run Shestakov example')

    # output files to plot
    parser.add_argument('outfiles', type=str, nargs='+',
                        help='output files to pliot')

    # output options
    parser.add_argument('--rthresh', type=float, default=1.0e-10,
                        help='residual to print threshold for each method')

    parser.add_argument('--relative', action='store_true',
                        help='plot relative iteration counts')

    parser.add_argument('--save', action='store_true',
                        help='save figure to file')

    # plot options
    parser.add_argument('--norm', type=str, default='RMS',
                        choices=['L2', 'RMS', 'Max'],
                        help='norm to use in plots')

    # debugging options
    parser.add_argument('--debug', action='store_true',
                        help='enable debugging output')

    # parse command line args
    args = parser.parse_args()

    # create figure and axes
    fig, ax = plt.subplots()

    rvals = list()
    cvals = list()

    if args.debug:
        print(args.outfiles)

    # find values for rows and columns
    for outfile in args.outfiles:

        if args.debug:
            print(outfile)

        # parse file name to get run settings
        fname = os.path.basename(outfile).split("_")

        # get method name and parameters, set title
        if "kinsol" in fname[0]:
            method = "KINSOL"
            power = fname[2]
            alpha = fname[4]
            beta = fname[6]
            mAA = fname[8]
            delayAA = fname[10]

            if args.debug:
                print(beta)
                print(mAA)

            if beta not in rvals:
                rvals.append(beta)
            if mAA not in cvals:
                cvals.append(mAA)

        elif "tango" in fname[0]:
            method = "Tango"
            power = fname[2]
            alpha = fname[4]
            beta = fname[6]

            if args.debug:
                print(alpha)

            if alpha not in rvals:
                rvals.append(alpha)
            if 0 not in cvals:
                cvals.append(0)

        else:
            print('ERROR: Unknown method')
            sys.exit()

    if args.debug:
        print(rvals)
        print(cvals)

    if method == "KINSOL":

        prefix = 'kinsol: '
        prefix = prefix + 'p = ' + str(power)
        prefix = prefix + ' delay = ' + str(delayAA)
        prefix = prefix + ' threshold = ' + str(args.rthresh)

        figname = 'kinsol'
        figname += '_p_' + str(power)
        figname += '_delay_' + str(delayAA)
        figname += '_thresh_' + str(args.rthresh)
        figname += '.pdf'

    else:

        prefix = 'Tango: '
        prefix = prefix + 'p = ' + str(power)

        figname = 'Tango'
        figname += '_p_' + str(power)
        figname += '.pdf'


    # ensure row and column values are sorted
    rvals.sort()
    cvals.sort()

    table = np.empty([len(rvals), len(cvals)])
    table[:] = np.nan

    if args.debug:
        print(table)

    # iterate over files and add to figure
    for outfile in args.outfiles:

        print(outfile)

        # parse file name to get run settings
        fname = os.path.basename(outfile).split("_")

        # get method name and parameters, set title
        if "kinsol" in fname[0]:
            method = "KINSOL"
            power = fname[2]
            alpha = fname[4]
            beta = fname[6]
            mAA = fname[8]
            delayAA = fname[10]

            ridx = find_first(rvals, beta)
            cidx = find_first(cvals, mAA)

        elif "tango" in fname[0]:
            method = "Tango"
            power = fname[2]
            alpha = fname[4]
            beta = fname[6]

            ridx = find_first(rvals, alpha)
            cidx = find_first(cvals, 0)

        else:
            print('ERROR: Unknown method')
            sys.exit()

        if args.debug:
            print(ridx, cidx)

        # load data
        data = np.loadtxt(outfile)

        # number of rows
        nrows = np.shape(data)[0]

        # array of iteration numbers
        iters = range(0, nrows)

        # compute norm of data
        if data.ndim > 1:
            nrm = np.zeros((nrows, 1))
            if args.norm == 'L2':
                for i in iters:
                    nrm[i] = np.sqrt(np.sum(data[i, :]**2))   # L2
            elif args.norm == 'RMS':
                for i in iters:
                    nrm[i] = np.sqrt(np.mean(data[i, :]**2))  # RMS
            else:
                for i in iters:
                    nrm[i] = np.amax(np.abs(data[i, :]))      # Max
        else:
            nrm = data

        # print out the first iteration below a given threshold
        # or the final iteratio if the threshold is not crossed
        for i in iters:
            if nrm[i] < args.rthresh:
                table[ridx][cidx] = i
                break

    if args.debug:
        print(table)

    miniter = np.nanmin(table)
    print(miniter)
    if args.relative:
        table = table / miniter

    cmap = mpl.cm.coolwarm
    # bounds = miniter * np.linspace(0.1, 2.0, num=20)
    bounds = miniter * np.linspace(1.0, 2.0, num=20)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='max')

    im = heatmap(table, rvals, cvals, ax=ax,
                 cmap=cmap, norm=norm)

    if args.relative:
        annotate_heatmap(im, valfmt="{x:.1f}", fontsize="xx-small")
    else:
        annotate_heatmap(im, valfmt="{x:.0f}", fontsize="xx-small")

    # hm = heatmap2(table, rvals, cvals, ax=ax)

    plt.title(prefix)

    if method == "KINSOL":
        ax.set_ylabel("beta")
    else:
        ax.set_ylabel("alpha")
    ax.set_xlabel("m")
    ax.xaxis.set_label_position('top')

    fig.tight_layout()

    if args.save:
        plt.savefig(figname, bbox_inches='tight')
    else:
        plt.show()


# ****** run main ****** #
if __name__ == '__main__':
    import sys
    sys.exit(main())
