#!/usr/bin/env python3
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

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tango import derivatives
from tango import HToMatrixFD
from tango import lodestro_method

import kinsol as kin

# slightly modified version of tango's noisyflux
from tango.extras import noisyflux_mod as noisyflux


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

# ****** Problem (Shestakov) ***** #
class Problem:

    def setup(args):

        Problem.args = args

        # problem setup
        L = 1                             # size of domain
        Problem.N = args.N                # number of spatial grid points
        Problem.dx = L / (Problem.N - 1)  # spatial grid size

        # location corresponding to grid points j=0, ..., N-1
        Problem.x = np.arange(Problem.N) * Problem.dx

        # boundary condition at x = L
        Problem.nL = 1e-2

        # time step  (effectively infinite)
        Problem.dt = 1e4

        # instantiate flux model
        Problem.fluxModel = FluxModel(Problem.dx, p=args.p)

        Problem.profile_ss = steady_state_solution(Problem.x, Problem.nL, p=args.p)
        Problem.flux_ss = Problem.fluxModel.get_flux(Problem.profile_ss)

        # initial condition
        if args.initial_condition == 'linear':
            # line between true solution at left and right boundary
            m = (Problem.profile_ss[-1] - Problem.profile_ss[0]) / L
            Problem.n_IC = m * Problem.x + Problem.profile_ss[0]
        elif args.initial_condition == 'orig':
            # original initial condition
            Problem.n_IC = 0.02 - 0.01 * Problem.x
        else:
            print("ERROR: Unknown initial condition!")
            sys.exit()

        # add noise to flux
        if args.noise:
            Problem.fluxModel = noisyflux.NoisyFlux(Problem.fluxModel,
                                                    args.noise_amplitude,
                                                    args.noise_lac,
                                                    Problem.dx)

        # initialize data storage
        Problem.residual_history = np.empty(args.max_iterations)
        Problem.error_history = np.empty(args.max_iterations)

        Problem.residual_history[:] = np.NaN
        Problem.error_history[:] = np.NaN

        if args.makeplots:
            Problem.profile_history = np.zeros((args.max_iterations + 1, Problem.N))
            Problem.flux_history = np.zeros((args.max_iterations + 1, Problem.N))

            Problem.profile_history[:,:] = np.NaN
            Problem.flux_history[:,:] = np.NaN

        # initialize FluxSplitter.
        # for many problems, the exact value of these parameters doesn't matter too much.
        #  these parameters have to do with the splitting between diffusive and convective flux.
        thetaParams = {'Dmin': 1e-5, 'Dmax': 1e13, 'dpdxThreshold': 10}

        Problem.fluxSplitter = lodestro_method.FluxSplit(thetaParams)

        Problem.iter_idx = 0

        Problem.flux = np.zeros_like(Problem.x)


    def Gfun_p(sunvec_old, sunvec_new, user_data):

        try:
            # extract old profile and flux
            state_old = kin.N_VGetData(sunvec_old)
            profile = state_old[:Problem.N]

            # get turbulent flux
            flux = Problem.fluxModel.get_flux(profile)

            # relax flux
            if Problem.iter_idx == 0:
                Problem.flux = flux
            else:
                Problem.flux = (1 - Problem.args.beta) * Problem.flux + Problem.args.beta * flux

            # transform flux into effective transport coefficients.  H2=D, H3=-c
            # [use flux split class from lodestro_method]
            (D, c, _) = Problem.fluxSplitter.flux_to_transport_coeffs(Problem.flux,
                                                                      profile,
                                                                      Problem.dx)

            H2Turb = D
            H3 = -c

            # get H's for all the others (H1, H2, H7).  H's represent terms in the transport equation
            H1 = np.ones_like(Problem.x)
            H7 = source(Problem.x)
            H2const = 0.00  # could represent some background level of (classical) diffusion
            H2 = H2Turb + H2const

            ## new --- discretize, then compute the residual, then solve the matrix equation for the new profile
            (A, B, C, f) = HToMatrixFD.H_to_matrix(Problem.dt,
                                                   Problem.dx,
                                                   Problem.nL,
                                                   Problem.n_IC,
                                                   H1, H2=H2, H3=H3, H7=H7)

            # see fieldgroups.calculate_residual() for additional information on the residual calculation
            resid = A*np.concatenate((profile[1:], np.zeros(1))) + B*profile + C*np.concatenate((np.zeros(1), profile[:-1])) - f
            resid = resid / np.max(np.abs(f))  # normalize the residual
            rmsResid = np.sqrt( np.mean( resid**2 ))  # take an rms characterization of residual
            Problem.residual_history[Problem.iter_idx] = rmsResid

            # error
            Problem.error_history[Problem.iter_idx] = np.max(np.abs(profile - Problem.profile_ss))

            # save data
            if Problem.args.makeplots:
                Problem.profile_history[Problem.iter_idx, :] = profile
                Problem.flux_history[Problem.iter_idx, :] = Problem.flux

            # solve matrix equation for new profile
            profile_new = HToMatrixFD.solve(A, B, C, f)

            # relax profile
            profile_new = (1 - Problem.args.alpha) * profile + Problem.args.alpha * profile_new

            # check
            if np.any(profile_new < 0) == True:
                print(f'error.  negative value detected in profile at iter = {Problem.iter_idx}')
                raise ValueError

            # extract old profile and flux
            state_new = kin.N_VGetData(sunvec_new)
            state_new[:Problem.N] = profile_new

            # update iteration count
            Problem.iter_idx += 1

        except:
            print(f'error.  Gfun_p failed at iter = {Problem.iter_idx}')
            return -1
        else:
            return 0


    def Gfun_pq(sunvec_old, sunvec_new, user_data):

        try:
            # extract old profile and flux
            state_old = kin.N_VGetData(sunvec_old)
            profile = state_old[:Problem.N]
            flux = state_old[Problem.N:]

            # transform flux into effective transport coefficients.  H2=D, H3=-c
            # [use flux split class from lodestro_method]
            (D, c, _) = Problem.fluxSplitter.flux_to_transport_coeffs(flux,
                                                                      profile,
                                                                      Problem.dx)

            H2Turb = D
            H3 = -c

            # get H's for all the others (H1, H2, H7).  H's represent terms in the transport equation
            H1 = np.ones_like(Problem.x)
            H7 = source(Problem.x)
            H2const = 0.00  # could represent some background level of (classical) diffusion
            H2 = H2Turb + H2const

            ## new --- discretize, then compute the residual, then solve the matrix equation for the new profile
            (A, B, C, f) = HToMatrixFD.H_to_matrix(Problem.dt,
                                                   Problem.dx,
                                                   Problem.nL,
                                                   Problem.n_IC,
                                                   H1, H2=H2, H3=H3, H7=H7)

            # see fieldgroups.calculate_residual() for additional information on the residual calculation
            resid = A*np.concatenate((profile[1:], np.zeros(1))) + B*profile + C*np.concatenate((np.zeros(1), profile[:-1])) - f
            resid = resid / np.max(np.abs(f))  # normalize the residual
            rmsResid = np.sqrt( np.mean( resid**2 ))  # take an rms characterization of residual
            Problem.residual_history[Problem.iter_idx] = rmsResid

            # error
            Problem.error_history[Problem.iter_idx] = np.max(np.abs(profile - Problem.profile_ss))

            # save data
            if Problem.args.makeplots:
                Problem.profile_history[Problem.iter_idx, :] = profile
                Problem.flux_history[Problem.iter_idx, :] = flux

            # solve matrix equation for new profile
            profile_new = HToMatrixFD.solve(A, B, C, f)

            # relax profile
            profile_new = (1 - Problem.args.alpha) * profile + Problem.args.alpha * profile_new

            # check
            if np.any(profile_new < 0) == True:
                print(f'error.  negative value detected in profile at iter = {Problem.iter_idx}')
                raise ValueError

            # compute new flux with relaxed flux
            flux_new = Problem.fluxModel.get_flux(profile_new)

            # relax flux
            flux_new = (1 - Problem.args.beta) * flux + Problem.args.beta * flux_new

            # extract old profile and flux
            state_new = kin.N_VGetData(sunvec_new)
            state_new[:Problem.N] = profile_new
            state_new[Problem.N:] = flux_new

            # update iteration count
            Problem.iter_idx += 1

        except:
            print(f'error.  Gfun_pq failed at iter = {Problem.iter_idx}')
            return -1
        else:
            return 0


    def Gfun_d(sunvec_old, sunvec_new, user_data):

        try:
            # extract old profile and flux
            state_old = kin.N_VGetData(sunvec_old)
            profile = state_old[:Problem.N]

            # get turbulent flux
            flux = Problem.fluxModel.get_flux(profile)

            # transform flux into effective transport coefficients.  H2=D, H3=-c
            # [use flux split class from lodestro_method]
            (D, c, _) = Problem.fluxSplitter.flux_to_transport_coeffs(flux,
                                                                      profile,
                                                                      Problem.dx)

            # compute relaxation of D, c
            # (EWMA = Exponentially Weighted Moving Average)
            if Problem.iter_idx == 0:
                Problem.D_EWMA = D
                Problem.c_EWMA = c
            else:
                Problem.D_EWMA = (Problem.args.gamma * D +
                                  (1 - Problem.args.gamma) * Problem.D_EWMA)
                Problem.c_EWMA = (Problem.args.gamma * c +
                                  (1 - Problem.args.gamma) * Problem.c_EWMA)

            H2Turb = Problem.D_EWMA
            H3 = -Problem.c_EWMA

            # get H's for all the others (H1, H2, H7).  H's represent terms in the transport equation
            H1 = np.ones_like(Problem.x)
            H7 = source(Problem.x)
            H2const = 0.00  # could represent some background level of (classical) diffusion
            H2 = H2Turb + H2const

            ## new --- discretize, then compute the residual, then solve the matrix equation for the new profile
            (A, B, C, f) = HToMatrixFD.H_to_matrix(Problem.dt,
                                                   Problem.dx,
                                                   Problem.nL,
                                                   Problem.n_IC,
                                                   H1, H2=H2, H3=H3, H7=H7)

            # see fieldgroups.calculate_residual() for additional information on the residual calculation
            resid = A*np.concatenate((profile[1:], np.zeros(1))) + B*profile + C*np.concatenate((np.zeros(1), profile[:-1])) - f
            resid = resid / np.max(np.abs(f))  # normalize the residual
            rmsResid = np.sqrt( np.mean( resid**2 ))  # take an rms characterization of residual
            Problem.residual_history[Problem.iter_idx] = rmsResid

            # error
            Problem.error_history[Problem.iter_idx] = np.max(np.abs(profile - Problem.profile_ss))

            # save data
            if Problem.args.makeplots:
                Problem.profile_history[Problem.iter_idx, :] = profile
                Problem.flux_history[Problem.iter_idx, :] = flux

            # solve matrix equation for new profile
            profile_new = HToMatrixFD.solve(A, B, C, f)

            # check
            if np.any(profile_new < 0) == True:
                print(f'error.  negative value detected in profile at iter = {Problem.iter_idx}')
                raise ValueError

            # extract old profile and flux
            state_new = kin.N_VGetData(sunvec_new)
            state_new[:Problem.N] = profile_new

            # update iteration count
            Problem.iter_idx += 1

        except:
            print(f'error.  Gfun_p failed at iter = {Problem.iter_idx}')
            return -1
        else:
            return 0


    def solveKINSOL(state):

        # scaling arrays
        scale = np.ones_like(state)

        # create N_Vector objects
        sunvec_state = kin.N_VMake_Serial(state)
        sunvec_scale = kin.N_VMake_Serial(scale)

        # allocate memory for KINSOL
        kmem = kin.KINCreate()

        # set number of prior residuals used in Anderson acceleration
        if Problem.args.aa_m > 0:
            flag = kin.KINSetMAA(kmem, Problem.args.aa_m)
            if flag < 0:
                raise RuntimeError(f'KINSetMAA returned {flag}')

        # wrap the python system function so that it is callable from C
        if Problem.args.gfun == 'pq':
            sysfn = kin.WrapPythonSysFn(Problem.Gfun_pq)
        elif Problem.args.gfun == 'p':
            sysfn = kin.WrapPythonSysFn(Problem.Gfun_p)
        else:
            sysfn = kin.WrapPythonSysFn(Problem.Gfun_d)

        # initialize KINSOL
        flag = kin.KINInitPy(kmem, sysfn, sunvec_state)
        if flag < 0:
            raise RuntimeError(f'KINInitPy returned {flag}')

        # specify stopping tolerance based on residual
        flag = kin.KINSetFuncNormTol(kmem, 1.0e-12)
        if flag < 0:
            raise RuntimeError(f'KINSetFuncNormTol returned {flag}')

        # set the maximum number of iterations
        flag = kin.KINSetNumMaxIters(kmem, Problem.args.max_iterations)
        if flag < 0:
            raise RuntimeError(f'KINSetUseMaxIters returned {flag}')

        # ignore convergence test and run to max iterations
        flag = kin.KINSetUseMaxIters(kmem, 1)
        if flag < 0:
            raise RuntimeError(f'KINSetUseMaxIters returned {flag}')

        # return the newest iteration at end
        flag = kin.KINSetReturnNewest(kmem, 1)
        if flag < 0:
            raise RuntimeError(f'KINSetReturnNewest returned {flag}')

        # set Anderson acceleration delay
        if Problem.args.aa_delay > 0:
            flag = kin.KINSetDelayAA(kmem, Problem.args.aa_delay)
            if flag < 0:
                raise RuntimeError(f'KINSetDelayAA returned {flag}')

        # set fixed point and Anderson acceleration damping
        if Problem.args.aa_damping < 1.0:
            flag = kin.KINSetDampingFP(kmem, Problem.args.aa_damping)
            if flag < 0:
                raise RuntimeError(f'KINSetDampingFP returned {flag}')

            flag = kin.KINSetDampingAA(kmem, Problem.args.aa_damping)
            if flag < 0:
                raise RuntimeError(f'KINSetDampingAA returned {flag}')

        # set error log file
        # flag = kin.KINSetErrFilename(kmem, "kinsol_error.log")
        # if flag < 0:
        #     raise RuntimeError(f'KINSetErrFilename returned {flag}')

        # set info file
        # flag = kin.KINSetInfoFilename(kmem, "kinsol_info.log")
        # if flag < 0:
        #     raise RuntimeError(f'KINSetInfoFilename returned {flag}')

        # set info print level
        # flag = kin.KINSetPrintLevel(kmem, 2)
        # if flag < 0:
        #     raise RuntimeError(f'KINSetPrintLevel returned {flag}')

        # Call KINSOL to solve problem
        flag = kin.KINSol(kmem,          # KINSOL memory block
                          sunvec_state,  # initial guess; solution vector
                          kin.KIN_FP,    # global strategy choice
                          sunvec_scale,  # scaling vector for the variable
                          sunvec_scale)  # scaling vector for function values
        if flag < 0:
            raise RuntimeError(f'KINSol returned {flag}')
        elif flag > 0:
            print(f'KINSol returned {flag}')
        else:
            print('KINSol finished')


# **************************************** #


parser = argparse.ArgumentParser(description='Run Shestakov example')


parser.add_argument('--N', type=int, default=500,
                    help='Number of spatial mesh points')

parser.add_argument('--gfun', default='p', choices=['p', 'pq', 'd'],
                    help='Fixed-point function formulation')

parser.add_argument('--initial_condition', type=str, default='orig',
                    choices=['orig','linear'],
                    help='Initial condition')

parser.add_argument('--p', type=int, default=2,
                    help='Power for analytic flux')

parser.add_argument('--noise', action='store_true',
                    help='Add noise to flux values')

parser.add_argument('--noise_lac', type=float, default=0.2,
                    help='Correlation length of noise')

parser.add_argument('--noise_amplitude', type=float, default=1.0e-4,
                    help='Amplitude of noise')

parser.add_argument('--alpha', type=float, default=1.0,
                    help='Relaxation parameter for profile')

parser.add_argument('--beta', type=float, default=1.0,
                    help='Relaxation parameter for flux')

parser.add_argument('--gamma', type=float, default=1.0,
                    help='Relaxation parameter for transport coefficients')

parser.add_argument('--max_iterations', type=int, default=200,
                    help='Maximum number iterations')

parser.add_argument('--aa_m', type=int, default=0,
                    help='Anderson acceleration depth')

parser.add_argument('--aa_delay', type=int, default=0,
                    help='Anderson acceleration delay')

parser.add_argument('--aa_damping', type=float, default=1.0,
                    help='Anderson acceleration damping')

parser.add_argument('--plot_off', dest='makeplots', action='store_false',
                    help='disable all plot options')

parser.add_argument('--plot_iters', type=int, nargs=2, default=[0, 10],
                    help='Iteration range to plot (inclusive)')

# parse command line args
args = parser.parse_args()

# sanity check
if args.gfun != 'd' and args.gamma < 1.0:
    print(f"ERROR: gfun = {args.gfun} with gamma = {args.gamma}")
    sys.exit()

if args.gfun == 'd' and (args.alpha < 1.0 or args.beta < 1.0):
    print(f"ERROR: gfun = {args.gfun} with alpha = {args.alpha} and "
          f"beta = {args.beta}")
    sys.exit()

if ((args.alpha < 1.0 or args.beta < 1.0 or args.gamma < 1.0) and
    args.aa_damping < 1.0):
    print("=========================================================")
    print("WARNING: Mixing internal relaxation and Anderson damping!")
    print("=========================================================")

# setup the problem
Problem.setup(args)

# initial profile
profile = Problem.n_IC.copy()

if args.gfun == 'pq':
    # initial flux
    flux = Problem.fluxModel.get_flux(profile)

    # create state vector
    state = np.concatenate((profile, flux))
else:
    state = profile.copy()

# solve the problem
Problem.solveKINSOL(state)

# extract profile and flux
if args.gfun == 'pq':
    profile = state[:Problem.N]
    flux = state[Problem.N:]
else:
    profile = state[:Problem.N]
    flux = Problem.fluxModel.get_flux(profile)

# shortcut to iteration count
iter_idx = Problem.iter_idx

# compute the final residual and error

print("Profile damping (alpha):                ", args.alpha)
print("Flux damping (beta):                    ", args.beta)
print("Transport coefficient damping (gamma):  ", args.gamma)
print("Acceleration depth:                     ", args.aa_m)
print("Acceleration delay:                     ", args.aa_delay)
print("Anderson damping:                       ", args.aa_damping)
print("Iterations:                             ", iter_idx)
print("Residual:                               ", Problem.residual_history[iter_idx - 1])
print("Error:                                  ", Problem.error_history[iter_idx - 1])

if ((args.alpha < 1.0 or args.beta < 1.0 or args.gamma < 1.0) and
    args.aa_damping < 1.0):
    print("=========================================================")
    print("WARNING: Mixing internal relaxation and Anderson damping!")
    print("=========================================================")

# plot the last iteration of density --- presumably the correct solution, if converged
#  Also plot the analytic steady state solution
if args.makeplots:

    profile_final = profile
    iters = np.arange(0, args.max_iterations)

    # save data final profile and flux
    Problem.profile_history[iter_idx, :] = profile
    Problem.flux_history[iter_idx, :] = flux

    # plot profile
    plt.figure()
    plt.plot(Problem.x, profile_final, 'b-', label='numerical solution')
    plt.plot(Problem.x, Problem.profile_ss, 'k--', label='analytic solution')
    plt.xlabel('x')
    plt.title('profile')
    plt.legend(loc='best')
    plt.grid()

    # plot residuals (excludes final value)
    plt.figure()
    plt.semilogy(iters, Problem.residual_history)
    plt.xlabel('iteration number')
    plt.title('Residual')
    plt.grid()

    # plot errors (excludes final value)
    plt.figure()
    plt.semilogy(iters, Problem.error_history)
    plt.xlabel('iteration number')
    plt.title('Max Error')
    plt.grid()

    # plot solution history (includes final value)
    fig, ax = plt.subplots()
    for i in range(args.plot_iters[0], args.plot_iters[1] + 1):
        plt.plot(Problem.x, Problem.profile_history[i,:], label=f'Iter {i}')
    plt.plot(Problem.x, Problem.profile_ss, 'k--', label='analytic solution')
    plt.xlabel('x')
    plt.title('Solution history')
    # place legend outside plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.grid()

    # plot flux history
    fig, ax = plt.subplots()
    for i in range(args.plot_iters[0], args.plot_iters[1] + 1):
        plt.plot(Problem.x, Problem.flux_history[i,:], label=f'Iter {i}')
    plt.plot(Problem.x, Problem.flux_ss, 'k--', label='analytic flux')
    plt.xlabel('x')
    plt.title('Flux history')
    # place legend outside plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.grid()

    # show plots
    plt.show()


# save the residual and error history
residual_error_history = np.vstack((Problem.residual_history, Problem.error_history))

outdir = "output_kinsol_gfun-" + args.gfun
if args.noise:
    outdir += '_add-noise'
else:
    outdir += '_no-noise'
outdir += '_p_' + str(args.p)
outdir += '_N_' + str(args.N)
outdir += '_IC_' + args.initial_condition
outdir += '_maxit_' + str(args.max_iterations)

if not os.path.exists(outdir):
    os.makedirs(outdir)

prefix = 'kinsol_gfun-' + args.gfun
if args.noise:
    prefix += '_add-noise'
else:
    prefix += '_no-noise'
prefix += '_p_' + str(args.p)
prefix += '_alpha_' + str(args.alpha)
prefix += '_beta_' + str(args.beta)
prefix += '_gamma_' + str(args.gamma)
prefix += '_m_' + str(args.aa_m)
prefix += '_delay_' + str(args.aa_delay)
prefix += '_damp_' + str(args.aa_damping)

filename = prefix + '_residual_error_history.txt'

np.savetxt(os.path.join(outdir, filename), residual_error_history)
