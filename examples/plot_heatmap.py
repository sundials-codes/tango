#!/usr/bin/env python3


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

    parser.add_argument('--title', type=str,
                        default="Anderson Acceleration Convergence Heatmap",
                        help='Heatmap title')

    parser.add_argument('--rowlabel', type=str, default="m",
                        help='Heatmap row label')

    parser.add_argument('--collabel', type=str, default="Damping",
                        help='Heatmap column label')

    parser.add_argument('--fontsize', type=str, default=12,
                        help='''Either an relative value of 'xx-small',
                        'x-small', 'small', 'medium', 'large', 'x-large',
                        'xx-large' or an absolute font size, e.g., 12''')

    parser.add_argument('--save', action='store_true',
                        help='Save figure to file')

    parser.add_argument('--figname', type=str,
                        help='Figure file name')

    # debugging options
    parser.add_argument('--debug', action='store_true',
                        help='enable debugging output')

    # parse command line args
    args = parser.parse_args()

    # remove duplicate files
    args.outfiles = list(set(args.outfiles))

    rows = list()
    cols = list()

    # find values for rows and columns
    for outfile in args.outfiles:

        if args.debug:
            print(outfile)

        # parse file name to get run settings
        fname = os.path.basename(outfile).split("_")

        if args.debug:
            for i, f in enumerate(fname):
                print(i, f)

        gfun = fname[1]
        noise = fname[2]
        power = int(fname[4])
        alpha = float(fname[6])
        beta = float(fname[8])
        gamma = float(fname[10])
        aa_m = int(fname[12])
        aa_delay = int(fname[14])
        aa_damp = float(fname[16])

        # update row labels
        rows.append(aa_m)

        # update column labels
        if alpha < 1.0 and beta < 1.0:
            if alpha != beta:
                print(f"ERROR: alpha = {alpha} != beta = {beta}")
                sys.exit()

        if ((alpha < 1.0 or beta < 1.0 or gamma < 1.0) and
            aa_damp < 1.0):
            print(f"ERROR: Mixed internal and Anderson damping not supported!")
            sys.exit()

        if ((alpha < 1.0 or beta < 1.0 or aa_damp < 1.0) and
            gamma < 1.0):
            print(f"ERROR: Mixed D/c and state/flux damping not supported!")
            sys.exit()

        if alpha < 1.0:
            cols.append(alpha)
        elif beta < 1.0:
            cols.append(beta)
        elif gamma < 1.0:
            cols.append(gamma)
        elif aa_damp < 1.0:
            cols.append(aa_damp)
        else:
            print(f"ERROR: No damping!")
            sys.exit()

    # remove duplicates
    rows = list(set(rows))
    rows.sort()

    cols = list(set(cols))
    cols.sort()

    # dictionary for row index
    row_map = dict()
    for i, a in enumerate(rows):
        row_map[a] = i

    # dictionary for column index
    col_map = dict()
    for i, c in enumerate(cols):
        col_map[c] = i

    # create output matrix
    data_out = np.empty((len(rows), len(cols)))
    data_out[:,:] = np.NaN

    # read data from files
    # find values for rows and columns
    for outfile in args.outfiles:

        if args.debug:
            print(outfile)

        # parse file name to get run settings
        fname = os.path.basename(outfile).split("_")

        gfun = fname[1]
        noise = fname[2]
        power = int(fname[4])
        alpha = float(fname[6])
        beta = float(fname[8])
        gamma = float(fname[10])
        aa_m = int(fname[12])
        aa_delay = int(fname[14])
        aa_damp = float(fname[16])

        if alpha < 1.0:
            col_idx = alpha
        elif beta < 1.0:
            col_idx = beta
        elif gamma < 1.0:
            col_idx = gamma
        elif aa_damp < 1.0:
            col_idx = aa_damp
        else:
            print(f"ERROR: No damping!")
            sys.exit()

        data_in = np.loadtxt(outfile)

        iters = np.shape(data_in)[1]

        # find the first iteration below a given threshold
        found = False
        for i in range(iters):
            if data_in[0][i] < args.rthresh:
                found = True
                idx = i
                if args.debug:
                    print(outfile, i, data_in[0][i])
                break

        # set output data
        if found:
            i = row_map[aa_m]
            j = col_map[col_idx]
            data_out[i][j] = idx

    print("    ", end='')
    for j in range(len(cols)):
        print(f"{cols[j]:8.2f}", end='')
    print()

    for i in range(len(rows)):
        print(f"{rows[i]:4d}", end='')
        for j in range(len(cols)):
            print(f"{data_out[i][j]:8.0f}", end='')
        print()

    # create figure and axes
    fig, ax = plt.subplots()

    miniter = np.nanmin(data_out)
    bounds = miniter * np.linspace(1.0, 2.0, num=20)

    cmap = mpl.cm.coolwarm
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='max')

    ims = plot_heatmap(rows, cols, data_out, ax=ax,
                       cmap=cmap, norm=norm)

    for im in ims:
        annotate_heatmap(im, valfmt="{x:.0f}", fontsize=args.fontsize)

    plt.title(args.title)
    ax.set_ylabel(args.rowlabel)
    ax.set_xlabel(args.collabel)
    ax.xaxis.set_label_position('top')

    fig.tight_layout()

    if alpha < 1.0 and beta < 1.0:
        damping_type = 'alpha-beta'
    elif alpha < 1.0:
        damping_type = 'alpha'
    elif beta < 1.0:
        damping_type = 'beta'
    elif gamma < 1.0:
        damping_type = 'gamma'
    elif aa_damp < 1.0:
        damping_type = 'aa-damping'
    else:
        damping_type = 'None'

    if args.figname:
        figname = args.figname
    else:
        figname = (f"residual_heatmap_{gfun}_{noise}_p_{power}"
                   f"_{damping_type}_delay_{aa_delay}.pdf")

    if args.save:
        plt.savefig(figname, bbox_inches='tight')
    else:
        plt.show()


def plot_heatmap(row_labels, col_labels, data, ax=None, **kwargs):

    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec

    if not ax:
        fig, ax = plt.subplots()

    # columnwise heatmap
    premask = np.tile(np.arange(data.shape[1]), data.shape[0]).reshape(data.shape)
    images = []
    for i in range(data.shape[1]):
        col = np.ma.array(data, mask = premask != i)
        im = ax.imshow(col, **kwargs)
        images.append(im)


    # Show all x and y ticks
    ax.set_xticks(np.arange(data.shape[1])) # cols
    ax.set_yticks(np.arange(data.shape[0])) # rows

    # Labels the ticks with the respective list entries
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Horizontal axis label appears on top
    ax.tick_params(bottom=False, top=False,
                   labelbottom=False, labeltop=True, left=False)

    # Turn spines off and create white grid
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=4)
    ax.tick_params(which="minor", bottom=False, left=False)

    return images


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

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

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

            if not isinstance(data[i,j], float):
                continue

            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


# ****** run main ****** #
if __name__ == '__main__':
    import sys
    sys.exit(main())
