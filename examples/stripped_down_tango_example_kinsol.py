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

import kinsol as kin

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
    def __init__(self, dx, p=2, firstOrderEdge=True):
        self.dx = dx
        self.p = p
        self.firstOrderEdge = firstOrderEdge
    def get_flux(self, profile):
        # Return flux Gamma on the same grid as n
        n = profile
        if self.firstOrderEdge:
            dndx = derivatives.dx_centered_difference_edge_first_order(n, self.dx)
        else:
            dndx = derivatives.dx_centered_difference(n, self.dx)
        Deff = np.abs( (dndx/n)**self.p )
        Gamma = -Deff * dndx
        return Gamma


# ****** Problem (Shestakov) ***** #
class Problem:

    def setup(args):

        #### create stuff

        # problem setup
        L  = args.L   # size of domain
        N  = args.N   # number of spatial grid points
        n0 = args.n0  # boundary value at 0
        p  = args.p   # power for analytic flux

        Problem.dx    = L / (N - 1)                # spatial grid size
        Problem.x     = np.arange(N) * Problem.dx  # location corresponding to grid points j=0, ..., N-1
        Problem.nL    = args.nL                    # boundary condition at L
        Problem.dt    = args.dt                    # time step  (1e4 is effectively infinite)
        Problem.alpha = args.alpha                 # relaxation parameter on the effective diffusion coefficient

        Problem.numGEvals = 0  # number of G evaluations
        Problem.numIters  = 0  # number of fixed point iterations

        # instantiate flux model
        Problem.fluxModel = FluxModel(Problem.dx, p=p, firstOrderEdge=args.firstOrderEdge)

        # initialize FluxSplitter.
        # for many problems, the exact value of these parameters doesn't matter too much.
        #  these parameters have to do with the splitting between diffusive and convective flux.
        thetaParams = {'Dmin': args.Dmin, 'Dmax': args.Dmax, 'dpdxThreshold': args.dpdxThreshold}

        Problem.fluxSplitter = lodestro_method.FluxSplit(thetaParams)

        # old time profile and initial guess
        n_IC = np.zeros_like(Problem.x)
        if args.IC == 'pow':
            # power law initial condition (q = 1 does not satisfy left BC)
            q = args.q
            n_IC[:] = ((Problem.nL - n0) / L**q) * Problem.x**q + n0
        elif args.IC == 'const':
            # constant initial condition except at the right BC
            c = args.c
            n_IC[:-1] = c
            n_IC[-1]  = Problem.nL
        else:
            n_IC[:] = nss[:]

        # set for old time
        Problem.n_mminus1 = np.copy(n_IC)

        # compute the analytic steady state solution
        Problem.nss = steady_state_solution(Problem.x, Problem.nL, p=p, L=L)

        # print problem setup to screen
        print("Tango Shestakov Example:")
        print("  Domain size L        =", L)
        print("  Mesh points N        =", N)
        print("  Mesh spacing dx      =", Problem.dx)
        print("  Initial condition    =", args.IC)
        if args.IC == 'pow':
            print("  IC power             =", args.q)
        elif args.IC == 'const':
            print("  IC const             =", args.c)
        print("  Left boundary value  =", n0)
        print("  Right boundary value =", Problem.nL)
        print("  Time step size       =", Problem.dt)
        print("  1st order edge       =", args.firstOrderEdge)
        print("  D minimum            =", args.Dmin)
        print("  D maximum            =", args.Dmax)
        print("  dp/dx threshold      =", args.dpdxThreshold)
        print("  Flux power           =", p)
        print("  Relaxation alpha     =", Problem.alpha)
        print("  Relaxation beta      =", args.beta)
        print("  Max iterations       =", args.maxIterations)

        # initialize data storage for full history (initial to end)
        Problem.nAll   = np.zeros((args.maxIterations+1, N))
        Problem.errAll = np.zeros_like(Problem.nAll)

        # initialize data storage for history excluding last iteration (initial to end - 1)
        Problem.fluxAll   = np.zeros((args.maxIterations, N))
        Problem.DAll      = np.zeros_like(Problem.fluxAll)
        Problem.cAll      = np.zeros_like(Problem.fluxAll)
        Problem.D_EWMAAll = np.zeros_like(Problem.fluxAll)
        Problem.c_EWMAAll = np.zeros_like(Problem.fluxAll)
        Problem.residAll  = np.zeros_like(Problem.fluxAll)
        Problem.FAll      = np.zeros_like(Problem.fluxAll)

        Problem.wrmsResidHistory = np.zeros(args.maxIterations)


    def Gfun(profile_old):

        # get turbulent flux
        flux = Problem.fluxModel.get_flux(profile_old)

        # transform flux into effective transport coefficients.  H2=D, H3=-c
        # [use flux split class from lodestro_method]
        (D, c, _) = Problem.fluxSplitter.flux_to_transport_coeffs(flux, profile_old, Problem.dx)

        # compute relaxation of D, c  (EWMA = Exponentially Weighted Moving Average)
        if Problem.numGEvals == 0:
            Problem.D_EWMA = D
            Problem.c_EWMA = c
        else:
            Problem.D_EWMA = Problem.alpha * D + (1 - Problem.alpha) * Problem.D_EWMA
            Problem.c_EWMA = Problem.alpha * c + (1 - Problem.alpha) * Problem.c_EWMA

        H2Turb = Problem.D_EWMA
        H3 = -Problem.c_EWMA

        # get H's for all the others (H1, H2, H7).  H's represent terms in the transport equation
        H1 = np.ones_like(Problem.x)
        H7 = source(Problem.x)
        H2const = 0.00  # could represent some background level of (classical) diffusion
        H2 = H2Turb + H2const

        ## new --- discretize, then compute the residual, then solve the matrix equation for the new profile
        (A, B, C, f) = HToMatrixFD.H_to_matrix(Problem.dt, Problem.dx, Problem.nL, Problem.n_mminus1, H1, H2=H2, H3=H3, H7=H7)

        # see fieldgroups.calculate_residual() for additional information on the residual calculation
        resid = A*np.concatenate((profile_old[1:], np.zeros(1))) + B*profile_old + C*np.concatenate((np.zeros(1), profile_old[:-1])) - f
        wrmsResid = np.sqrt( np.mean( (resid / np.max(np.abs(f)))**2 ) )  # compute normalized rms norm

        # save data for plots
        Problem.fluxAll[Problem.numGEvals,:]   = flux
        Problem.DAll[Problem.numGEvals,:]      = D
        Problem.cAll[Problem.numGEvals,:]      = c
        Problem.D_EWMAAll[Problem.numGEvals,:] = Problem.D_EWMA
        Problem.c_EWMAAll[Problem.numGEvals,:] = Problem.c_EWMA
        Problem.residAll[Problem.numGEvals,:]  = resid

        Problem.wrmsResidHistory[Problem.numGEvals] = wrmsResid

        # solve matrix equation for new profile n_{i+1} = G(n_i)
        profile_new = HToMatrixFD.solve(A, B, C, f)

        # compute F_i = G(n_i) - n_i
        Problem.FAll[Problem.numGEvals, :] = profile_new - profile_old

        # update number of G evals
        Problem.numGEvals += 1

        return profile_new


    def solve(profile_old, maxIterations, beta):

        # create array for new profile
        profile_new = np.zeros_like(profile_old)

        # save the initial profile and error
        Problem.nAll[0, :]   = profile_old
        Problem.errAll[0, :] = profile_old - Problem.nss

        # perform fixed point iteration
        for iterationNumber in np.arange(0, maxIterations):

            # evaluate n_{i+1} = G(n_i)
            profile_new[:] = Problem.Gfun(profile_old)

            # relax profile
            profile_new[:] = beta * profile_new + (1.0 - beta) * profile_old

            # save new profile and compute new error
            Problem.nAll[iterationNumber + 1, :]   = profile_new
            Problem.errAll[iterationNumber + 1, :] = profile_new - Problem.nss

            # update iteration count
            Problem.numIters += 1

            # check
            if np.any(profile_old < 0) == True:
                print(f'error.  negative value detected in profile at l={iterationNumber}')
                break

            # make new profile old
            profile_old = np.copy(profile_new)

        return profile_new


    def GfunKINSOL(sunvec_profile_old, sunvec_profile_new, user_data):

        # extract arrays
        profile_old = kin.N_VGetData(sunvec_profile_old)
        profile_new = kin.N_VGetData(sunvec_profile_new)

        if Problem.numGEvals > 0:
            # save new profile and compute new error
            Problem.nAll[Problem.numGEvals, :]   = profile_old
            Problem.errAll[Problem.numGEvals, :] = profile_old - Problem.nss

        profile_new[:] = Problem.Gfun(profile_old)

        # update iteration count
        Problem.numIters += 1

        return 0


    def solveKINSOL(profile_old, maxIterations, tol=1.0e-11, beta=1.0, m=0, delay=0):

        # save the initial profile and error
        Problem.nAll[0, :]   = profile_old
        Problem.errAll[0, :] = profile_old - Problem.nss

        # solution and scaling arrays
        profile_new = np.copy(profile_old)
        scale       = np.ones_like(profile_old)

        # create N_Vector objects
        sunvec_profile = kin.N_VMake_Serial(profile_new)
        sunvec_scale   = kin.N_VMake_Serial(scale)

        # allocate memory for KINSOL
        kmem = kin.KINCreate()

        # set number of prior residuals used in Anderson acceleration
        if m > 0:
            flag = kin.KINSetMAA(kmem, m)
            if flag < 0: raise RuntimeError(f'KINSetMAA returned {flag}')

        # wrap the python system function so that it is callable from C
        sysfn = kin.WrapPythonSysFn(Problem.GfunKINSOL)

        # initialize KINSOL
        flag = kin.KINInitPy(kmem, sysfn, sunvec_profile)
        if flag < 0: raise RuntimeError(f'KINInitPy returned {flag}')

        # specify stopping tolerance based on residual
        flag = kin.KINSetFuncNormTol(kmem, tol)
        if flag < 0: raise RuntimeError(f'KINSetFuncNormTol returned {flag}')

        # ignore convergence test and run to max iterations
        flag = kin.KINSetUseMaxIters(kmem, 1)
        if flag < 0: raise RuntimeError(f'KINSetUseMaxIters returned {flag}')

        # return the newest iteration at end
        flag = kin.KINSetReturnNewest(kmem, 1)
        if flag < 0: raise RuntimeError(f'KINSetReturnNewest returned {flag}')

        # set Anderson acceleration delay
        if delay > 0:
            flag = kin.KINSetDelayAA(kmem, delay)
            if flag < 0: raise RuntimeError(f'KINSetDelayAA returned {flag}')

        # set fixed point and Anderson acceleration damping
        if beta < 1.0:
            flag = kin.KINSetDampingFP(kmem, beta)
            if flag < 0: raise RuntimeError(f'KINSetDampingFP returned {flag}')

            flag = kin.KINSetDampingAA(kmem, beta)
            if flag < 0: raise RuntimeError(f'KINSetDampingAA returned {flag}')

        # set error log file
        flag = kin.KINSetErrFilename(kmem, "kinsol_error.log")
        if flag < 0: raise RuntimeError(f'KINSetErrFilename returned {flag}')

        # set info file
        flag = kin.KINSetInfoFilename(kmem, "kinsol_info.log")
        if flag < 0: raise RuntimeError(f'KINSetInfoFilename returned {flag}')

        # set info print level
        flag = kin.KINSetPrintLevel(kmem, 2)
        if flag < 0: raise RuntimeError(f'KINSetPrintLevel returned {flag}')

        # Call KINSOL to solve problem
        flag = kin.KINSol(kmem,            # KINSOL memory block
                          sunvec_profile,  # initial guess on input; solution vector
                          kin.KIN_FP,      # global strategy choice
                          sunvec_scale,    # scaling vector for the variable cc
                          sunvec_scale)    # scaling vector for function values fval
        if flag < 0:
            raise RuntimeError(f'KINSol returned {flag}')
        elif flag > 0:
            print(f'KINSol returned {flag}')
        else:
            print('KINSol finished')

        # save final profile and error
        Problem.nAll[Problem.numGEvals, :]   = profile_new
        Problem.errAll[Problem.numGEvals, :] = profile_new - Problem.nss

        # Print solution and solver statistics
        flag, fnorm = kin.KINGetFuncNorm(kmem)
        if flag < 0: raise RuntimeError(f'KINGetFuncNorm returned {flag}')

        print('Computed solution (||F|| = %Lg):\n' % fnorm)

        # Free memory
        #kin.KINFree(kmem)
        #kin.N_VDestroy(sunvec_profile)
        #kin.N_VDestroy(sunvec_scale)

        return profile_new


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
    parser.add_argument('--centerdiff', dest='firstOrderEdge', action='store_false',
                        help='use second order center differences in FluxModel')

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
    parser.add_argument('--maxIterations', type=int, default=200,
                        help='maximum number iterations')

    # KINSOL options
    parser.add_argument('--kinsol', action='store_true',
                        help='solve with KINSOL')
    parser.add_argument('--mAA', type=int, default=0,
                        help='Anderson acceleration depth')
    parser.add_argument('--delayAA', type=int, default=0,
                        help='number of iterations to delay Anderson acceleration')

    # norm option (only for plots right now since iteration always runs to max)
    parser.add_argument('--norm', type=str, default='RMS',
                        choices=['L2','RMS','Max'],
                        help='norm to use in plots')

    #  plotting options
    parser.add_argument('--plotall', action='store_true',
                        help='enable all plot options')
    parser.add_argument('--noplots', dest='makeplots', action='store_false',
                        help='disable all plot options')
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
    parser.add_argument('--noref', dest='plotref', action='store_false',
                        help='disable convergence rate reference line')
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

    # setup the problem
    Problem.setup(args)

    # solve the problem
    nInitial = np.copy(Problem.n_mminus1)

    if args.kinsol:
        nFinal = Problem.solveKINSOL(nInitial, args.maxIterations, beta=args.beta,
                                     m=args.mAA, delay=args.delayAA)
    else:
        nFinal = Problem.solve(nInitial, args.maxIterations, args.beta)

    # print final resiudal and error
    print("Finished:")
    print("  Interations =", Problem.numIters)
    if args.norm == 'L2':
        res_nrm = np.sqrt(np.sum(Problem.residAll[-1,:]**2))
        err_nrm = np.sqrt(np.sum(Problem.errAll[-1,:]**2))
    elif args.norm == 'RMS':
        res_nrm = np.sqrt(np.mean(Problem.residAll[-1,:]**2))
        err_nrm = np.sqrt(np.mean(Problem.errAll[-1,:]**2))
    else:
        res_nrm = np.amax(np.abs(Problem.residAll[-1,:]))
        err_nrm = np.amax(np.abs(Problem.errAll[-1,:]))
    print("  Residual (" + args.norm + " norm) =", res_nrm)
    print("  Error    (" + args.norm + " norm) =", err_nrm)

    # iteration range to plot
    iters   = np.arange(0, Problem.numIters)     # initial to end - 1 (length numIters)
    itersp1 = np.arange(0, Problem.numIters + 1) # initial to end     (length numIters + 1)

    # write history to file

    outdir = args.outputdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # add a prefix for different configurations
    if args.kinsol:
        prefix = 'kinsol'
        prefix = prefix + '_alpha_' + str(args.alpha)
        prefix = prefix + '_beta_' + str(args.beta)
        prefix = prefix + '_m_' + str(args.mAA)
        prefix = prefix + '_delay_' + str(args.delayAA)
    else:
        prefix = 'tango'
        prefix = prefix + '_alpha_' + str(args.alpha)
        prefix = prefix + '_beta_' + str(args.beta)

    # full history
    np.savetxt(outdir + '/' + prefix + '_n_history.txt',   Problem.nAll)
    np.savetxt(outdir + '/' + prefix + '_err_history.txt', Problem.errAll)

    # up to but not including last iteration
    np.savetxt(outdir + '/' + prefix + '_flux_history.txt',   Problem.fluxAll)
    np.savetxt(outdir + '/' + prefix + '_D_history.txt',      Problem.DAll)
    np.savetxt(outdir + '/' + prefix + '_c_history.txt',      Problem.cAll)
    np.savetxt(outdir + '/' + prefix + '_D_EWMA_history.txt', Problem.D_EWMAAll)
    np.savetxt(outdir + '/' + prefix + '_c_EWMA_history.txt', Problem.c_EWMAAll)
    np.savetxt(outdir + '/' + prefix + '_resid_history.txt',  Problem.residAll)

    # final plots

    if args.makeplots:

        if (args.plotall or args.plotfinalsolution):

            # plot final solution
            plt.figure()
            plt.plot(Problem.x, nFinal, label='numerical solution')
            plt.plot(Problem.x, Problem.nss, 'k--', label='analytic solution')
            plt.xlabel('x')
            plt.ylabel('n')
            plt.title('Final Solution')
            plt.legend(loc='best')
            plt.grid()

        if (args.plotall or args.plotconvergence):

            # plot residual norm history
            res_nrm = np.zeros((Problem.numIters,1))
            for i in iters:
                if args.norm == 'L2':
                    res_nrm[i] = np.sqrt(np.sum(Problem.residAll[i,:]**2))  # L2
                elif args.norm == 'RMS':
                    res_nrm[i] = np.sqrt(np.mean(Problem.residAll[i,:]**2)) # RMS
                else:
                    res_nrm[i] = np.amax(np.abs(Problem.residAll[i,:]))     # Max

            # convergence rate reference line
            if args.plotref:
                # estimate convergence constant
                idx = min(args.refidx[0], Problem.numIters - 1)
                c   = res_nrm[idx] / res_nrm[idx-1]

                # min to cutoff ref plot
                min_ref = np.amin(res_nrm) / 2.0

                # create convergence rate reference
                res_ref = np.zeros((Problem.numIters,1))
                res_ref[0] = (5.0 * res_nrm[idx]) / c**idx
                plt_idx = -1
                for i in range(1, Problem.numIters):
                    res_ref[i] = c * res_ref[i-1]
                    if res_ref[i] < min_ref:
                        plt_idx = i
                        break

            plt.figure()
            plt.semilogy(iters, res_nrm, nonpositive='clip', label='residual')
            if args.plotref:
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
            resF_nrm = np.zeros((Problem.numIters, 1))
            for i in iters:
                if args.norm == 'L2':
                    resF_nrm[i] = np.sqrt(np.sum(Problem.FAll[i,:]**2))  # L2
                elif args.norm == 'RMS':
                    resF_nrm[i] = np.sqrt(np.mean(Problem.FAll[i,:]**2)) # RMS
                else:
                    resF_nrm[i] = np.amax(np.abs(Problem.FAll[i,:]))     # Max

            # convergence rate reference line
            if args.plotref:
                # estimate convergence constant
                idx = min(args.refidx[1], Problem.numIters - 1)
                c   = resF_nrm[idx] / resF_nrm[idx-1]

                # min to cutoff ref plot
                min_ref = np.amin(resF_nrm) / 2.0

                # create convergence rate reference
                resF_ref = np.zeros((Problem.numIters,1))
                resF_ref[0] = (5.0 * resF_nrm[idx]) / c**idx
                plt_idx = -1
                for i in range(1, Problem.numIters):
                    resF_ref[i] = c * resF_ref[i-1]
                    if (resF_ref[i] < min_ref):
                        plt_idx = i
                        break

            plt.figure()
            plt.semilogy(iters, resF_nrm, nonpositive='clip', label='residual')
            if args.plotref:
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
            err_nrm = np.zeros((Problem.numIters + 1, 1))
            for i in itersp1:
                if args.norm == 'L2':
                    err_nrm[i] = np.sqrt(np.sum(Problem.errAll[i,:]**2))  # L2
                elif args.norm == 'RMS':
                    err_nrm[i] = np.sqrt(np.mean(Problem.errAll[i,:]**2)) # RMS
                else:
                    err_nrm[i] = np.amax(np.abs(Problem.errAll[i,:]))     # Max

            # convergence rate reference line
            if args.plotref:
                # estimate convergence constant
                idx = min(args.refidx[2], Problem.numIters - 1)
                c   = err_nrm[idx] / err_nrm[idx-1]

                # min to cutoff ref plot
                min_ref = np.amin(err_nrm) / 2.0

                # create convergence rate reference
                err_ref = np.zeros((Problem.numIters + 1,1))
                err_ref[0] = (5.0 * err_nrm[idx]) / c**idx
                plt_idx = -1
                for i in range(1, Problem.numIters + 1):
                    err_ref[i] = c * err_ref[i-1]
                    if (err_ref[i] < min_ref):
                        plt_idx = i
                        break

            plt.figure()
            plt.semilogy(itersp1, err_nrm, nonpositive='clip', label='residual')
            # convergence rate reference line
            if args.plotref:
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

        if (args.plotall or args.plotfinalreserr):

            # plot final absolute residual
            res = np.abs(Problem.residAll[-1,:])
            plt.figure()
            plt.semilogy(Problem.x, res)
            plt.xlabel('x')
            plt.ylabel('$\|R\|$')
            plt.title('Final Absolute Residual')
            plt.grid()

            # plot final absolute F residual
            resF = np.abs(Problem.FAll[-1,:])
            plt.figure()
            plt.semilogy(Problem.x, resF)
            plt.xlabel('x')
            plt.ylabel('$\|F_i = G(n_i) - n_i\|$')
            plt.title('Final Absolute F Resiudal')
            plt.grid()

            # plot final absolute error
            err = np.abs(Problem.errAll[-1,:])
            plt.figure()
            plt.semilogy(Problem.x, err)
            plt.xlabel('x')
            plt.ylabel('$\|n - n_{ss}\|$')
            plt.title('Final Absolute Error')
            plt.grid()


        # history range to plot
        min_iter = max(args.historyrange[0], 0)
        max_iter = min(args.historyrange[1], args.maxIterations)
        num_iter = max_iter - min_iter + 1;

        # set color cycle history plot lines
        if (num_iter > 20):
            mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.plasma(np.linspace(0, 1, num_iter)))
        elif (num_iter > 10):
            mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.tab20(np.linspace(0, 1, num_iter)))
        else:
            mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.tab10(np.linspace(0, 1, num_iter)))

        if (args.plotall or args.plotsolutionhistory):

            # plot solution history
            fig, ax = plt.subplots()
            for i in range(min_iter, max_iter + 1):
                ax.plot(Problem.x, Problem.nAll[i], label=i)
            plt.plot(Problem.x, Problem.nss, 'k--', label='analytic solution')
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            plt.xlabel('x')
            plt.ylabel('n')
            plt.title('Solution History')
            plt.grid()

        if (args.plotall or args.plotreserrhistory):

            # plot residual history
            fig, ax = plt.subplots()
            for i in range(min_iter, min(max_iter + 1, args.maxIterations)):
                ax.semilogy(Problem.x, np.abs(Problem.residAll[i]), label=i)
            ax.semilogy(Problem.x, np.abs(Problem.residAll[args.maxIterations - 1]), 'k--', label='final')
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            plt.xlabel('x')
            plt.ylabel('$\|R\|$')
            plt.title('Absolute Residual History')
            plt.grid()

            # plot F residual history
            fig, ax = plt.subplots()
            for i in range(min_iter, min(max_iter + 1, args.maxIterations)):
                ax.semilogy(Problem.x, np.abs(Problem.FAll[i]), label=i)
            ax.semilogy(Problem.x, np.abs(Problem.FAll[args.maxIterations - 1]), 'k--', label='final')
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            plt.xlabel('x')
            plt.ylabel('$\|F_i = G(n_i) - n_i\|$')
            plt.title('Absolute F Residual History')
            plt.grid()

            # plot absolute error history
            fig, ax = plt.subplots()
            for i in range(min_iter, max_iter + 1):
                ax.semilogy(Problem.x, np.abs(Problem.errAll[i]), label=i)
            ax.semilogy(Problem.x, np.abs(Problem.errAll[args.maxIterations]), 'k--', label='final')
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            plt.xlabel('x')
            plt.ylabel('$\|n - n_{ss}\|$')
            plt.title('Absolute Error History')
            plt.grid()

        if (args.plotall or args.plotfluxdchistory):

            # plot flux history
            fig, ax = plt.subplots()
            for i in range(min_iter, min(max_iter + 1, args.maxIterations)):
                ax.semilogy(Problem.x, Problem.fluxAll[i], label=i)
            ax.semilogy(Problem.x, Problem.fluxAll[args.maxIterations - 1], 'k--', label='final')
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            plt.xlabel('x')
            plt.ylabel('flux')
            plt.title('Flux History')
            plt.grid()

            # plot D history
            fig, ax = plt.subplots()
            for i in range(min_iter, min(max_iter + 1, args.maxIterations)):
                ax.semilogy(Problem.x, Problem.DAll[i], label=i)
            ax.semilogy(Problem.x, Problem.DAll[args.maxIterations - 1], 'k--', label='final')
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            plt.xlabel('x')
            plt.ylabel('D')
            plt.title('D History')
            plt.grid()

            # plot D_EWMA history
            fig, ax = plt.subplots()
            for i in range(min_iter, min(max_iter + 1, args.maxIterations)):
                ax.semilogy(Problem.x, Problem.D_EWMAAll[i], label=i)
            ax.semilogy(Problem.x, Problem.D_EWMAAll[args.maxIterations - 1], 'k--', label='final')
            ax.semilogy(Problem.x, Problem.DAll[args.maxIterations - 1], 'r:', label='final (no relaxation)')
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            plt.xlabel('x')
            plt.ylabel('D')
            plt.title('Relaxed D History')
            plt.grid()

            # plot c history
            fig, ax = plt.subplots()
            for i in range(min_iter, min(max_iter + 1, args.maxIterations)):
                ax.semilogy(Problem.x, Problem.cAll[i], label=i)
            ax.semilogy(Problem.x, Problem.cAll[args.maxIterations - 1], 'k--', label='final')
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            plt.xlabel('x')
            plt.ylabel('c')
            plt.title('c History')
            plt.grid()

            # plot c_EMWA history
            fig, ax = plt.subplots()
            for i in range(min_iter, min(max_iter + 1, args.maxIterations)):
                ax.semilogy(Problem.x, Problem.c_EWMAAll[i], label=i)
            ax.semilogy(Problem.x, Problem.c_EWMAAll[args.maxIterations - 1], 'k--', label='final')
            ax.semilogy(Problem.x, Problem.cAll[args.maxIterations - 1], 'r:', label='final (no relaxation)')
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            plt.xlabel('x')
            plt.ylabel('c')
            plt.title('Relaxed c History')
            plt.grid()

        # display all plots
        if (args.plotall or
            args.plotfinalsolution or
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
