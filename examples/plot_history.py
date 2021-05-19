#!/usr/bin/env python

# ****** Main ***** #
def main():

    import argparse
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Run Shestakov example')

    # output files to plot
    parser.add_argument('outfiles', type=str, nargs='+',
                        help='Output files to pliot')

    parser.add_argument('--maxiter', type=int,
                        help='Max iteration to plot')

    parser.add_argument('--row', type=int, default=0,
                        help='Which row to plots')

    parser.add_argument('--title', type=str,
                        help='Plot title')

    parser.add_argument('--xlabel', type=str, default="Iteration",
                        help='Plot x-axis label')

    # parser.add_argument('--ylabel', type=str,
    #                     default='$||\frac{M_i p_i - g}{max(|g|)}||_{rms}$',
    #                     help='Plot y-axis label')

    parser.add_argument('--ylabel', type=str,
                        default='RMS Norm',
                        help='Plot y-axis label')

    parser.add_argument('--legend', type=str, nargs='+',
                        help='Legend entries')

    parser.add_argument('--legendtitle', type=str,
                        help='Legend Title')

    parser.add_argument('--save', action='store_true',
                        help='Save figure to file')

    parser.add_argument('--figname', type=str,
                        help='Figure file name')

    # debugging options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debugging output')

    # parse command line args
    args = parser.parse_args()

    # create figure and axes
    fig, ax = plt.subplots()

    # iterate over files and add to figure
    for i, outfile in enumerate(args.outfiles):

        if args.debug:
            print(i, outfile)

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

        # load data
        data = np.loadtxt(outfile)

        # number of columns
        num_iters = np.shape(data)[1]

        if args.maxiter:
            num_iters = min(args.maxiter, num_iters)

        iters = range(num_iters)

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

        if args.legend:
            label = args.legend[i]
        else:
            if damping_type == 'alpha-beta':
                label = f"{alpha:.2f}"
            elif damping_type == 'alpha':
                label = f"{alpha:.2f}"
            elif damping_type == 'beta':
                label = f"{beta:.2f}"
            elif damping_type == 'gamma':
                label = f"{gamma:.2f}"
            elif damping_type == 'aa-damping':
                label = f"{aa_damp:.2f}"
            else:
                label = None

        # plot method convergence
        ax.semilogy(iters, data[args.row,:num_iters], nonpositive='clip', label=label)

    ax.set_ylabel(args.ylabel)
    ax.set_xlabel(args.xlabel)

    if args.legendtitle:
        legend_title = args.legendtitle
    else:
        legend_title = "Damping"

    # place legend outside figure
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0),
              title=legend_title)
    #ax.legend(loc='best', title="Damping")

    if args.title:
        title = args.title
    else:
        title = "Residual History"

    plt.title(title)

    # add grid
    plt.grid()

    fig.tight_layout()

    if args.figname:
        figname = args.figname
    else:
        figname = (f"residual_history_{gfun}_{noise}_p_{power}"
                   f"_{damping_type}_m_{aa_m}_delay_{aa_delay}"
                   f".pdf")

    if args.save:
        plt.savefig(figname, bbox_inches='tight')
    else:
        plt.show()


# ****** run main ****** #
if __name__ == '__main__':
    import sys
    sys.exit(main())
