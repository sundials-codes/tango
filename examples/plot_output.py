#!/usr/bin/env python

# ****** Main ***** #
def main():

    import argparse

    import numpy as np
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Run Shestakov example')

    # output files to plot
    parser.add_argument('outfiles', type=str, nargs='+',
                        help='output files to pliot')

    # output options
    parser.add_argument('--rthresh', type=float, default=1.0e-10,
                        help='residual to print threshold for each method')

    # plot options
    parser.add_argument('--maxiter', type=int, default=None,
                        help='''max iteration value for stopping convergence
                        lines''')
    parser.add_argument('--cutoff', type=float, default=None,
                        help='threshold for stopping convergence lines')
    parser.add_argument('--norm', type=str, default='RMS',
                        choices=['L2', 'RMS', 'Max'],
                        help='norm to use in plots')
    parser.add_argument('--title', type=str, default=None,
                        help='set the plot title')
    parser.add_argument('--legend', type=str, nargs='+',
                        help='set legend entries')
    parser.add_argument('--legendinfo', type=str, nargs='+',
                        choices=['method', 'alpha', 'beta', 'mAA', 'delayAA'],
                        help='information to include in the legend')
    parser.add_argument('--legendoutside', action='store_true',
                        help='place legend outside plot')

    # reference line options
    parser.add_argument('--refidx', type=int, default=0,
                        help='''which output file (number) to base refine line
                        on''')
    parser.add_argument('--noref', dest='plotref', action='store_false',
                        help='disable convergence rate reference line')
    parser.add_argument('--ithresh', type=float, default=1.0e-14,
                        help='''threshold for stopping convergence reference
                        lines''')
    parser.add_argument('--cthresh', type=float, default=1.0e-8,
                        help='''threshold for estimating 1st order convergence
                        constant''')

    # debugging options
    parser.add_argument('--debug', action='store_true',
                        help='enable debugging output')

    # parse command line args
    args = parser.parse_args()

    # iterate over output files and add to plot
    fcount = 0

    # create figure and axes
    fig, ax = plt.subplots()

    # iterate over files and add to figure
    for outfile in args.outfiles:

        # parse file name to get run settings
        fname = outfile.split("_")

        # get method name and parameters, set title
        if "kinsol" in fname[0]:
            method = "KINSOL"
            alpha = fname[2]
            beta = fname[4]
            mAA = fname[6]
            delayAA = fname[8]

            # create legend entry for this data
            if args.legend:
                legend = args.legend[fcount]
            elif args.legendinfo:
                legend = make_legend_label(args.legendinfo, method, alpha,
                                           beta, mAA, delayAA)
            else:
                legend = None

        elif "tango" in fname[0]:
            method = "Tango"
            alpha = fname[2]
            beta = fname[4]

            # create legend entry for this data
            if args.legend:
                legend = args.legend[fcount]
            elif args.legendinfo:
                legend = make_legend_label(args.legendinfo, method, alpha,
                                           beta)
            else:
                legend = None

        else:
            print('ERROR: Unknown method')
            sys.exit()

        # load data
        data = np.loadtxt(outfile)

        # number of rows
        nrows = np.shape(data)[0]

        # array of iteration numbers
        iters = range(0, nrows)

        # compute norm of data
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

        nrm_pltidx = nrows + 1
        if args.cutoff:
            for i in iters:
                if (nrm[i] < args.cutoff):
                    nrm_pltidx = i + 1
                    break

        if args.maxiter:
            nrm_pltidx = min(nrm_pltidx, args.maxiter)

        # plot method convergence
        ax.semilogy(iters[:nrm_pltidx], nrm[:nrm_pltidx], nonpositive='clip',
                    label=legend)

        # print out the first iteration below a given threshold
        # or the final iteratio if the threshold is not crossed
        found = False
        for i in iters:
            if nrm[i] < args.rthresh:
                found = True
                print(fcount, i, nrm[i])
                break
        if not found:
            print(fcount, iters[-1], nrm[-1])

        # convergence rate reference line
        if args.plotref and fcount == args.refidx:

            # threshold for stopping ref plot
            min_ref = max(np.amin(nrm) / 2.0, args.ithresh)
            if args.cutoff:
                min_ref = max(min_ref, args.cutoff)

            # threshold for esitmating convergence constant
            idx = nrows
            for i in iters:
                if nrm[i] < args.cthresh:
                    idx = i - 1
                    break

            # estimate the 1st order convergence constant
            c_1 = nrm[idx] / nrm[idx-1]

            # create 1st order convergence rate reference
            ref = np.zeros((nrows, 1))
            ref[0] = (5.0 * nrm[idx]) / c_1**idx
            ref_pltidx = -1
            for i in range(1, nrows):
                ref[i] = c_1 * ref[i-1]
                if (ref[i] < min_ref):
                    ref_pltidx = i
                    break

        # update file count
        fcount += 1

    # add reference line last
    if args.plotref:
        ax.semilogy(iters[:ref_pltidx], ref[:ref_pltidx],
                    'k--', nonpositive='clip', label='1st order')

    # create plot
    plt.xlabel('Iteration')
    if args.norm == 'L2':
        plt.ylabel('$||F_i = G(n_i) - n_i||_{L2}$')
    elif args.norm == 'RMS':
        plt.ylabel('$||F_i = G(n_i) - n_i||_{RMS}$')
    else:
        plt.ylabel('$||F_i = G(n_i) - n_i||_{max}$')
    if args.title:
        plt.title(args.title)
    else:
        plt.title('Residual History')
    if args.legendinfo or args.plotref:
        if (args.legendoutside):
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        else:
            ax.legend(loc='best')
    plt.grid()
    plt.show()


# ****** create legend label ****** #
def make_legend_label(entries, method, alpha, beta, mAA=None, delayAA=None):

    # number of legend elements
    nlegend = len(entries)

    # just method name or paramter values
    if nlegend == 1:

        if entries[0] == 'method':
            legend = method
        elif entries[0] == 'alpha':
            legend = r'$\alpha = {0}$'.format(alpha)
        elif entries[0] == 'beta':
            legend = r'$\beta = {0}$'.format(beta)
        elif entries[0] == 'mAA' and mAA:
            legend = r'$m_{{AA}} = {0}$'.format(mAA)
        elif entries[0] == 'delayAA' and delayAA:
            legend = r'$d_{{AA}} = {0}$'.format(delayAA)

    # include paramter names before values
    else:

        legend = ''
        for i in range(nlegend):
            if i == 1:
                legend = legend + ' '

            if entries[i] == 'method':
                legend = legend + method
            elif entries[i] == 'alpha':
                legend = legend + r'$\alpha = {0}$'.format(alpha)
            elif entries[i] == 'beta':
                legend = legend + r'$\beta = {0}$'.format(beta)
            elif entries[i] == 'mAA' and mAA:
                legend = legend + r'$m_{{AA}} = {0}$'.format(mAA)
            elif entries[i] == 'delayAA' and delayAA:
                legend = legend + r'$d_{{AA}} = {0}$'.format(delayAA)

    return legend


# ****** run main ****** #
if __name__ == '__main__':
    import sys
    sys.exit(main())
