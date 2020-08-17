#!/usr/bin/env python
"""
stripped_down_tango_example.py

Example for how to use tango to solve a turbulence and transport problem.

Here, the "turbulent flux" is specified analytically, using the example (slightly modified and generalized) in the Shestakov et al. (2003) paper.
This example is a nonlinear diffusion equation with specified diffusion coefficient and source.  There is a closed form answer for the steady
state solution which can be compared with the numerically found solution.
"""

## Adding the Tango directory to the PYTHONPATH environment variable is recommended.
##  But if tango is not added to the PYTHONPATH, these commands can be used to add them manually.
#import sys
#sys.path.append("/path/to/tango")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from tango import derivatives
from tango import HToMatrixFD
from tango import lodestro_method


# ****** Solution ****** #
def steady_state_solution(x, nL, p=2, S0=1, delta=0.1, L=1):
    """Return the exact steady state solution for the Shestakov test problem

    Inputs:
      x             Spatial coordinate grid (array)
      nL            boundary condition n(L) (scalar)
      p             parameter for power exponent in Shestakov diffusion (scalar)
      S0            parameter in source term --- amplitude (scalar)
      delta         parameter in source term --- location where it turns off (scalar)
      L             parameter for size of domain (scalar)
    Outputs:
    """
    a = 1 / (p+1)   # convenient shortcut

    nright = ( nL**a  +  a * (S0 * delta)**a * (L-x) )**(p+1)
    nleft = ( nL**a  +  a * (S0 * delta)**a * (L - delta + (p+1)/(p+2) * delta * (1 - (x/delta)**((p+2)/(p+1)))))**(p+1)
    nss = nright
    nss[x < delta] = nleft[x < delta]
    return nss


# ****** Source ****** #
def source(x, S0=1):
    """Source that is nonzero for xa <= x <= xb.
    Return the source S."""
    S = np.zeros_like(x)
    xa = 0.0
    xb = 0.1
    S[(x >= xa) & (x <= xb)] = S0
    return S


# ****** Flux Model (Shestakov) ***** #
class FluxModel:
    def __init__(self, dx, p=2):
        self.dx = dx
        self.p = p
    def get_flux(self, profile):
        # Return flux Gamma on the same grid as n
        n = profile
        dndx = derivatives.dx_centered_difference_edge_first_order(n, self.dx)
        Deff = np.abs( (dndx/n)**self.p )
        Gamma = -Deff * dndx
        return Gamma


# ****** Main ***** #
def main():

    import os
    import argparse

    parser = argparse.ArgumentParser(description='Run Shestakov example')



    # problem setup options
    parser.add_argument('--p', type=float, default=2.0,
                        help='power for analytic flux')
    parser.add_argument('--L', type=float, default=1.0,
                        help='domain size [0,L]')
    parser.add_argument('--N', type=int, default=500,
                        help='number of spatial grid points')
    parser.add_argument('--n0', type=float, default=2.0e-2,
                        help='boundary value at x = 0 (left)')
    parser.add_argument('--nL', type=float, default=1.0e-2,
                        help='boundary value at x = L (right)')
    parser.add_argument('--IC', type=str, default='pow',
                        choices=['pow', 'const', 'sol'],
                        help='set initial condition type')
    parser.add_argument('--c', type=float, default=1.0,
                        help='value for constant IC')
    parser.add_argument('--q', type=float, default=1.0,
                        help='power in pow IC')
    parser.add_argument('--dt', type=float, default=1e4,
                        help='time step size')

    # flux splitter options
    parser.add_argument('--Dmin', type=float, default=1e-5,
                        help='Minimum D value')
    parser.add_argument('--Dmax', type=float, default=1e13,
                        help='Maximum D value')
    parser.add_argument('--dpdxThreshold', type=float, default=10,
                        help='dpdx threshold value')

    # relaxation and iteration options
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Relaxation parameter for diffusion')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Relaxation parameter for profile')
    parser.add_argument('--maxiters', type=int, default=200,
                        help='maximum number iterations')

    # norm option (only for plots right now since iteration always runs to max)
    parser.add_argument('--norm', type=str, default='RMS',
                        choices=['L2','RMS','Max'],
                        help='norm to use in plots')

    #  plotting options
    parser.add_argument('--nosolutionplot', dest='plotfinalsolution', action='store_false',
                        help='disable final solution plot')
    parser.add_argument('--noconvplot', dest='plotconvergence', action='store_false',
                        help='disable convergence plots')
    parser.add_argument('--plotfinalreserr', action='store_true',
                        help='enable final residual and error plots')
    parser.add_argument('--plotsolutionhistory', action='store_true',
                        help='enable solution history plot')
    parser.add_argument('--plotreserrhistory', action='store_true',
                        help='enable residual and error history plots')
    parser.add_argument('--plotfluxdchistory', action='store_true',
                        help='enable flux, D, and c history plots')
    parser.add_argument('--historyrange', type=int, nargs=2, default=[0, 200],
                        help='iterations to use in history plots (inclusive range)')
    parser.add_argument('--refidx', type=int, nargs=3, default=[100, 50, 10],
                        help='index to use for convergence reference line (resid, F resid, err)')

    # output options
    parser.add_argument('--outputdir', type=str, default='output',
                        help='output directory')

    # debugging options
    parser.add_argument('--debug', action='store_true',
                        help='enable debugging output')


    # parse command line args
    args = parser.parse_args()

    #### create stuff
    maxIterations = args.maxiters
    numIters = maxIterations
    alpha = args.alpha  # relaxation parameter on the effective diffusion coefficient
    beta  = args.beta   # relaxation perameter on the profile
    p = args.p          # power for analytic flux

    # problem setup
    L = args.L          # size of domain
    N = args.N          # number of spatial grid points
    dx = L / (N-1)      # spatial grid size
    x = np.arange(N)*dx # location corresponding to grid points j=0, ..., N-1

    # boundary condition
    nL = args.nL
    n0 = args.n0

    # time step  (1e4 is effectively infinite)
    dt = args.dt

    # compute the analytic steady state solution
    nss = steady_state_solution(x, nL, p=p, L=L)

    # initial condition
    n_IC = np.zeros_like(x)
    if args.IC == 'pow':
        # power law initial condition (q = 1 does not satisfy left BC)
        q = args.q
        n_IC[:] = ((nL - n0) / L**q) * x**q + n0
    elif args.IC == 'const':
        # constant initial condition except at the right BC
        c = args.c
        n_IC[:-1] = c
        n_IC[-1]  = nL
    else:
        n_IC[:] = nss[:]

    if args.debug:
        # plot initial condition
        plt.figure()
        plt.plot(x, n_IC)
        plt.xlabel('x')
        plt.ylabel('n_IC')
        plt.title('Initial Condition')
        plt.grid()
        plt.show()

    # print problem setup to screen
    print("Tango Shestakov Example:")
    print("  Domain size L        =", L)
    print("  Mesh points N        =", N)
    print("  Mesh spacing dx      =", dx)
    print("  Initial condition    =", args.IC)
    if args.IC == 'pow':
        print("  IC power             =", args.q)
    elif args.IC == 'const':
        print("  IC const             =", args.c)
    print("  Left boundary value  =", n0)
    print("  Right boundary value =", nL)
    print("  Time step size       =", dt)
    print("  D minimum            =", args.Dmin)
    print("  D maximum            =", args.Dmax)
    print("  dp/dx threshold      =", args.dpdxThreshold)
    print("  Flux power           =", p)
    print("  Relaxation alpha     =", alpha)
    print("  Relaxation beta      =", beta)
    print("  Max iterations       =", maxIterations)

    # create and fill arrays for old time, old iteration, and current profile
    n_mminus1   = np.copy(n_IC)
    profile_old = np.copy(n_IC)
    profile     = np.copy(n_IC)

    # instantiate flux model
    fluxModel = FluxModel(dx, p=p)

    # initialize data storage for full history (initial to end)
    nAll   = np.zeros((maxIterations+1, N))
    errAll = np.zeros_like(nAll)

    # initialize data storage for history excluding last iteration (initial to end - 1)
    fluxAll   = np.zeros((maxIterations, N))
    DAll      = np.zeros_like(fluxAll)
    cAll      = np.zeros_like(fluxAll)
    D_EWMAAll = np.zeros_like(fluxAll)
    c_EWMAAll = np.zeros_like(fluxAll)
    residAll  = np.zeros_like(fluxAll)
    FAll      = np.zeros_like(fluxAll)

    wrmsResidHistory = np.zeros(maxIterations)

    # initialize FluxSplitter.
    # for many problems, the exact value of these parameters doesn't matter too much.
    #  these parameters have to do with the splitting between diffusive and convective flux.
    thetaParams = {'Dmin': args.Dmin, 'Dmax': args.Dmax, 'dpdxThreshold': args.dpdxThreshold}

    fluxSplitter = lodestro_method.FluxSplit(thetaParams)

    # save the initial profile and error
    nAll[0, :]   = profile
    errAll[0, :] = profile - nss

    for iterationNumber in np.arange(0, maxIterations):

        # get turbulent flux
        flux = fluxModel.get_flux(profile)

        # transform flux into effective transport coefficients.  H2=D, H3=-c
        # [use flux split class from lodestro_method]
        (D, c, _) = fluxSplitter.flux_to_transport_coeffs(flux, profile, dx)

        # compute relaxation of D, c  (EWMA = Exponentially Weighted Moving Average)
        if iterationNumber == 0:
            D_EWMA = D
            c_EWMA = c
        else:
            D_EWMA = alpha * D + (1 - alpha) * D_EWMA
            c_EWMA = alpha * c + (1 - alpha) * c_EWMA

        H2Turb = D_EWMA
        H3 = -c_EWMA

        # get H's for all the others (H1, H2, H7).  H's represent terms in the transport equation
        H1 = np.ones_like(x)
        H7 = source(x)
        H2const = 0.00  # could represent some background level of (classical) diffusion
        H2 = H2Turb + H2const

        ## new --- discretize, then compute the residual, then solve the matrix equation for the new profile
        (A, B, C, f) = HToMatrixFD.H_to_matrix(dt, dx, nL, n_mminus1, H1, H2=H2, H3=H3, H7=H7)

        # see fieldgroups.calculate_residual() for additional information on the residual calculation
        resid = A*np.concatenate((profile[1:], np.zeros(1))) + B*profile + C*np.concatenate((np.zeros(1), profile[:-1])) - f
        wrmsResid = np.sqrt( np.mean( (resid / np.max(np.abs(f)))**2 ) )  # compute normalized rms norm

        # save data for plots
        fluxAll[iterationNumber,:]   = flux
        DAll[iterationNumber,:]      = D
        cAll[iterationNumber,:]      = c
        D_EWMAAll[iterationNumber,:] = D_EWMA
        c_EWMAAll[iterationNumber,:] = c_EWMA
        residAll[iterationNumber,:]  = resid

        wrmsResidHistory[iterationNumber] = wrmsResid

        # save old profile
        profile_old[:] = profile[:]

        # solve matrix equation for new profile n_{i+1} = G(n_i)
        profile[:] = HToMatrixFD.solve(A, B, C, f)

        # compute F_i = G(n_i) - n_i
        FAll[iterationNumber, :] = profile - profile_old

        # relax profile
        profile = beta * profile + (1.0 - beta) * profile_old

        # save new profile and compute new error
        nAll[iterationNumber+1, :]   = profile
        errAll[iterationNumber+1, :] = profile - nss

        # check
        if np.any(profile < 0) == True:
            print(f'error.  negative value detected in profile at l={iterationNumber}')
            numIters=iterationNumber + 1
            break

    # finish

    nFinal = np.copy(profile)

    # print final resiudal and error
    print("Finished:")
    print("  Interations =", numIters)
    if args.norm == 'L2':
        res_nrm = np.sqrt(np.sum(residAll[-1,:]**2))
        err_nrm = np.sqrt(np.sum(errAll[-1,:]**2))
    elif args.norm == 'RMS':
        res_nrm = np.sqrt(np.mean(residAll[-1,:]**2))
        err_nrm = np.sqrt(np.mean(errAll[-1,:]**2))
    else:
        res_nrm = np.amax(np.abs(residAll[-1,:]))
        err_nrm = np.amax(np.abs(errAll[-1,:]))
    print("  Residual (" + args.norm + " norm) =", res_nrm)
    print("  Error    (" + args.norm + " norm) =", err_nrm)

    # iteration range to plot
    iters   = np.arange(0, numIters)     # initial to end - 1 (length numIters)
    itersp1 = np.arange(0, numIters + 1) # initial to end     (length numIters + 1)

    # write history to file

    outdir = args.outputdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # full history
    np.savetxt(outdir + '/n_history.txt', nAll)
    np.savetxt(outdir + '/err_history.txt', nAll)

    # up to but not including last iteration
    np.savetxt(outdir + '/flux_history.txt',   fluxAll)
    np.savetxt(outdir + '/D_history.txt',      DAll)
    np.savetxt(outdir + '/c_history.txt',      cAll)
    np.savetxt(outdir + '/D_EWMA_history.txt', D_EWMAAll)
    np.savetxt(outdir + '/c_EWMA_history.txt', c_EWMAAll)
    np.savetxt(outdir + '/resid_history.txt',  residAll)

    # final plots

    if (args.plotfinalsolution):

        # plot final solution
        plt.figure()
        plt.plot(x, nFinal, label='numerical solution')
        plt.plot(x, nss, 'k--', label='analytic solution')
        plt.xlabel('x')
        plt.ylabel('n')
        plt.title('Final Solution')
        plt.legend(loc='best')
        plt.grid()

    if (args.plotconvergence):

        # plot residual norm history
        res_nrm = np.zeros((numIters,1))
        for i in iters:
            if args.norm == 'L2':
                res_nrm[i] = np.sqrt(np.sum(residAll[i,:]**2))  # L2
            elif args.norm == 'RMS':
                res_nrm[i] = np.sqrt(np.mean(residAll[i,:]**2)) # RMS
            else:
                res_nrm[i] = np.amax(np.abs(residAll[i,:]))     # Max

        # estimate convergence constant
        idx = args.refidx[0]
        c   = res_nrm[idx] / res_nrm[idx-1]

        # min to cutoff ref plot
        min_ref = np.amin(res_nrm) / 2.0

        # create convergence rate reference
        res_ref = np.zeros((numIters,1))
        res_ref[0] = (5.0 * res_nrm[idx]) / c**idx
        plt_idx = -1
        for i in range(1, numIters):
            res_ref[i] = c * res_ref[i-1]
            if res_ref[i] < min_ref:
                plt_idx = i
                break

        plt.figure()
        plt.semilogy(iters, res_nrm, nonpositive='clip', label='residual')
        plt.semilogy(iters[:plt_idx+1], res_ref[:plt_idx+1], 'k--', nonpositive='clip', label='1st order')

        plt.xlabel('Iteration')
        if args.norm == 'L2':
            plt.ylabel('$||R||_{L2}$')
        elif args.norm == 'RMS':
            plt.ylabel('$||R||_{RMS}$')
        else:
            plt.ylabel('$||R||_{max}$')
        plt.title('Residual History')
        plt.legend(loc='best')
        plt.grid()

        # plot F residual norm history
        resF_nrm = np.zeros((numIters, 1))
        for i in iters:
            if args.norm == 'L2':
                resF_nrm[i] = np.sqrt(np.sum(FAll[i,:]**2))  # L2
            elif args.norm == 'RMS':
                resF_nrm[i] = np.sqrt(np.mean(FAll[i,:]**2)) # RMS
            else:
                resF_nrm[i] = np.amax(np.abs(FAll[i,:]))     # Max

        # estimate convergence constant
        idx = args.refidx[1]
        c   = resF_nrm[idx] / resF_nrm[idx-1]

        # min to cutoff ref plot
        min_ref = np.amin(resF_nrm) / 2.0

        # create convergence rate reference
        resF_ref = np.zeros((numIters,1))
        resF_ref[0] = (5.0 * resF_nrm[idx]) / c**idx
        plt_idx = -1
        for i in range(1, numIters):
            resF_ref[i] = c * resF_ref[i-1]
            if (resF_ref[i] < min_ref):
                plt_idx = i
                break

        plt.figure()
        plt.semilogy(iters, resF_nrm, nonpositive='clip', label='residual')
        plt.semilogy(iters[:plt_idx+1], resF_ref[:plt_idx+1], 'k--', nonpositive='clip', label='1st order')
        plt.xlabel('Iteration')
        if args.norm == 'L2':
            plt.ylabel('$||F_i = G(n_i) - n_i||_{L2}$')
        elif args.norm == 'RMS':
            plt.ylabel('$||F_i = G(n_i) - n_i||_{RMS}$')
        else:
            plt.ylabel('$||F_i = G(n_i) - n_i||_{max}$')
        plt.title('F Residual History')
        plt.legend(loc='best')
        plt.grid()

        # plot error norm history
        err_nrm = np.zeros((numIters + 1, 1))
        for i in itersp1:
            if args.norm == 'L2':
                err_nrm[i] = np.sqrt(np.sum(errAll[i,:]**2))  # L2
            elif args.norm == 'RMS':
                err_nrm[i] = np.sqrt(np.mean(errAll[i,:]**2)) # RMS
            else:
                err_nrm[i] = np.amax(np.abs(errAll[i,:]))     # Max

        # estimate convergence constant
        idx = args.refidx[2]
        c   = err_nrm[idx] / err_nrm[idx-1]

        # min to cutoff ref plot
        min_ref = np.amin(err_nrm) / 2.0

        # create convergence rate reference
        err_ref = np.zeros((numIters + 1,1))
        err_ref[0] = (5.0 * err_nrm[idx]) / c**idx
        plt_idx = -1
        for i in range(1, numIters + 1):
            err_ref[i] = c * err_ref[i-1]
            if (err_ref[i] < min_ref):
                plt_idx = i
                break

        plt.figure()
        plt.semilogy(itersp1, err_nrm, nonpositive='clip', label='residual')
        plt.semilogy(itersp1[:plt_idx+1], err_ref[:plt_idx+1], 'k--', nonpositive='clip', label='1st order')
        plt.xlabel('Iteration')
        if args.norm == 'L2':
            plt.ylabel('$||n - n_{ss}||_{L2}$')
        elif args.norm == 'RMS':
            plt.ylabel('$||n - n_{ss}||_{RMS}$')
        else:
            plt.ylabel('$||n - n_{ss}||_{max}$')
        plt.title('Error History')
        plt.legend(loc='best')
        plt.grid()

    if (args.plotfinalreserr):

        # plot final absolute residual
        res = np.abs(residAll[-1,:])
        plt.figure()
        plt.semilogy(x, res)
        plt.xlabel('x')
        plt.ylabel('$\|R\|$')
        plt.title('Final Absolute Residual')
        plt.grid()

        # plot final absolute F residual
        resF = np.abs(FAll[-1,:])
        plt.figure()
        plt.semilogy(x, resF)
        plt.xlabel('x')
        plt.ylabel('$\|F_i = G(n_i) - n_i\|$')
        plt.title('Final Absolute F Resiudal')
        plt.grid()

        # plot final absolute error
        err = np.abs(errAll[-1,:])
        plt.figure()
        plt.semilogy(x, err)
        plt.xlabel('x')
        plt.ylabel('$\|n - n_{ss}\|$')
        plt.title('Final Absolute Error')
        plt.grid()


    # history range to plot
    min_iter = max(args.historyrange[0], 0)
    max_iter = min(args.historyrange[1], maxIterations)
    num_iter = max_iter - min_iter + 1;

    # set color cycle history plot lines
    if (num_iter > 20):
        mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.plasma(np.linspace(0, 1, num_iter)))
    elif (num_iter > 10):
        mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.tab20(np.linspace(0, 1, num_iter)))
    else:
        mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.tab10(np.linspace(0, 1, num_iter)))

    if (args.plotsolutionhistory):

        # plot solution history
        fig, ax = plt.subplots()
        for i in range(min_iter, max_iter + 1):
            ax.plot(x, nAll[i], label=i)
        plt.plot(x, nss, 'k--', label='analytic solution')
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.xlabel('x')
        plt.ylabel('n')
        plt.title('Solution History')
        plt.grid()

    if (args.plotreserrhistory):

        # plot residual history
        fig, ax = plt.subplots()
        for i in range(min_iter, min(max_iter + 1, maxIterations)):
            ax.semilogy(x, np.abs(residAll[i]), label=i)
        ax.semilogy(x, np.abs(residAll[maxIterations - 1]), 'k--', label='final')
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.xlabel('x')
        plt.ylabel('$\|R\|$')
        plt.title('Absolute Residual History')
        plt.grid()

        # plot F residual history
        fig, ax = plt.subplots()
        for i in range(min_iter, min(max_iter + 1, maxIterations)):
            ax.semilogy(x, np.abs(FAll[i]), label=i)
        ax.semilogy(x, np.abs(FAll[maxIterations - 1]), 'k--', label='final')
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.xlabel('x')
        plt.ylabel('$\|F_i = G(n_i) - n_i\|$')
        plt.title('Absolute F Residual History')
        plt.grid()

        # plot absolute error history
        fig, ax = plt.subplots()
        for i in range(min_iter, max_iter + 1):
            ax.semilogy(x, np.abs(errAll[i]), label=i)
        ax.semilogy(x, np.abs(errAll[maxIterations]), 'k--', label='final')
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.xlabel('x')
        plt.ylabel('$\|n - n_{ss}\|$')
        plt.title('Absolute Error History')
        plt.grid()

    if (args.plotfluxdchistory):

        # plot flux history
        fig, ax = plt.subplots()
        for i in range(min_iter, min(max_iter + 1, maxIterations)):
            ax.semilogy(x, fluxAll[i], label=i)
        ax.semilogy(x, fluxAll[maxIterations - 1], 'k--', label='final')
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.xlabel('x')
        plt.ylabel('flux')
        plt.title('Flux History')
        plt.grid()

        # plot D history
        fig, ax = plt.subplots()
        for i in range(min_iter, min(max_iter + 1, maxIterations)):
            ax.semilogy(x, DAll[i], label=i)
        ax.semilogy(x, DAll[maxIterations - 1], 'k--', label='final')
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.xlabel('x')
        plt.ylabel('D')
        plt.title('D History')
        plt.grid()

        # plot D_EWMA history
        fig, ax = plt.subplots()
        for i in range(min_iter, min(max_iter + 1, maxIterations)):
            ax.semilogy(x, D_EWMAAll[i], label=i)
        ax.semilogy(x, D_EWMAAll[maxIterations - 1], 'k--', label='final')
        ax.semilogy(x, DAll[maxIterations - 1], 'r:', label='final (no relaxation)')
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.xlabel('x')
        plt.ylabel('D')
        plt.title('Relaxed D History')
        plt.grid()

        # plot c history
        fig, ax = plt.subplots()
        for i in range(min_iter, min(max_iter + 1, maxIterations)):
            ax.semilogy(x, cAll[i], label=i)
        ax.semilogy(x, cAll[maxIterations - 1], 'k--', label='final')
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.xlabel('x')
        plt.ylabel('c')
        plt.title('c History')
        plt.grid()

        # plot c_EMWA history
        fig, ax = plt.subplots()
        for i in range(min_iter, min(max_iter + 1, maxIterations)):
            ax.semilogy(x, c_EWMAAll[i], label=i)
        ax.semilogy(x, c_EWMAAll[maxIterations - 1], 'k--', label='final')
        ax.semilogy(x, cAll[maxIterations - 1], 'r:', label='final (no relaxation)')
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.xlabel('x')
        plt.ylabel('c')
        plt.title('Relaxed c History')
        plt.grid()

    # display all plots
    if (args.plotfinalsolution or
        args.plotconvergence or
        args.plotfinalreserr or
        args.plotsolutionhistory or
        args.plotreserrhistory or
        args.plotfluxdchistory):

        # show plots
        plt.show()



# ****** run main ****** #
if __name__ == '__main__':
    import sys
    sys.exit(main())
