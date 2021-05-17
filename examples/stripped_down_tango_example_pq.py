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

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tango import derivatives
from tango import HToMatrixFD
from tango import lodestro_method

# slightly modified version of tango's noisyflux
import noisyflux_mod as noisyflux


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


# **************************************** #


parser = argparse.ArgumentParser(description='Run Shestakov example')

parser.add_argument('--p', type=int, default=2,
                    help='Power for analytic flux')

parser.add_argument('--noise', action='store_true',
                    help='Add noise to flux values')

parser.add_argument('--noise_lac', type=float, default=0.2,
                    help='Correlation length of noise')

parser.add_argument('--noise_amplitude', type=float, default=0.1,
                    help='Amplitude of noise')

parser.add_argument('--alpha', type=float, default=0.1,
                    help='Relaxation parameter for profile')

parser.add_argument('--beta', type=float, default=0.1,
                    help='Relaxation parameter for flux')

parser.add_argument('--max_iterations', type=int, default=200,
                    help='Maximum number iterations')

parser.add_argument('--plot_off', dest='makeplots', action='store_false',
                    help='disable all plot options')

parser.add_argument('--plot_iters', type=int, nargs=2, default=[0, 200],
                    help='Iteration range to plot (inclusive)')

# parse command line args
args = parser.parse_args()

# problem setup
L = 1                  # size of domain
N = 500                # number of spatial grid points
dx = L / (N-1)         # spatial grid size
x = np.arange(N) * dx  # location corresponding to grid points j=0, ..., N-1

# initial condition
n_IC = 0.02 - 0.01 * x

# boundary condition at x = L
nL = 1e-2

# time step  (effectively infinite)
dt = 1e4

# instantiate flux model
fluxModel = FluxModel(dx, p=args.p)

profile_ss = steady_state_solution(x, nL, p=args.p)
flux_ss = fluxModel.get_flux(profile_ss)

# add noise to flux
if args.noise:
    fluxModel = noisyflux.NoisyFlux(fluxModel,
                                    args.noise_amplitude,
                                    args.noise_lac,
                                    dx)

# initialize data storage
residual_history = np.zeros(args.max_iterations)
error_history = np.zeros(args.max_iterations)

residual_history[:] = np.NaN
error_history[:] = np.NaN

if args.makeplots:
    profile_history = np.zeros((args.max_iterations + 1, N))
    flux_history = np.zeros_like(profile_history)

    profile_history[:,:] = np.NaN
    flux_history[:,:] = np.NaN


# initialize FluxSplitter.
# for many problems, the exact value of these parameters doesn't matter too much.
#  these parameters have to do with the splitting between diffusive and convective flux.
thetaParams = {'Dmin': 1e-5, 'Dmax': 1e13, 'dpdxThreshold': 10}

fluxSplitter = lodestro_method.FluxSplit(thetaParams)

# initial profile
profile = n_IC.copy()

# initial flux
flux = fluxModel.get_flux(profile)

neg_profile = True

for iter_idx in np.arange(0, args.max_iterations):

    # transform flux into effective transport coefficients.  H2=D, H3=-c
    # [use flux split class from lodestro_method]
    (D, c, _) = fluxSplitter.flux_to_transport_coeffs(flux, profile, dx)

    H2Turb = D
    H3 = -c

    # get H's for all the others (H1, H2, H7).  H's represent terms in the transport equation
    H1 = np.ones_like(x)
    H7 = source(x)
    H2const = 0.00  # could represent some background level of (classical) diffusion
    H2 = H2Turb + H2const

    ## new --- discretize, then compute the residual, then solve the matrix equation for the new profile
    (A, B, C, f) = HToMatrixFD.H_to_matrix(dt, dx, nL, n_IC, H1, H2=H2, H3=H3, H7=H7)

    # see fieldgroups.calculate_residual() for additional information on the residual calculation
    resid = A*np.concatenate((profile[1:], np.zeros(1))) + B*profile + C*np.concatenate((np.zeros(1), profile[:-1])) - f
    resid = resid / np.max(np.abs(f))  # normalize the residual
    rmsResid = np.sqrt( np.mean( resid**2 ))  # take an rms characterization of residual
    residual_history[iter_idx] = rmsResid

    # error
    error_history[iter_idx] = np.max(np.abs(profile - profile_ss))

    # save data
    if args.makeplots:
        profile_history[iter_idx, :] = profile
        flux_history[iter_idx, :] = flux

    # solve matrix equation for new profile
    profile_new = HToMatrixFD.solve(A, B, C, f)

    # relax profile
    profile_new = (1 - args.alpha) * profile + args.alpha * profile_new

    # check
    if np.any(profile_new < 0) == True:
        neg_profile = False
        print(f'error.  negative value detected in profile at l={iter_idx}')
        break

    # compute new flux with relaxed flux
    flux_new = fluxModel.get_flux(profile_new)

    # relax flux
    flux_new = (1 - args.beta) * flux + args.beta * flux_new

    # update profle and flux
    profile = profile_new
    flux = flux_new


# update iteration count
iter_idx += 1

# compute the final residual and error

# finish

print("Iterations: ", iter_idx)
print("Residual:   ", residual_history[iter_idx - 1])
print("Error:      ", error_history[iter_idx - 1])

# plot the last iteration of density --- presumably the correct solution, if converged
#  Also plot the analytic steady state solution
if args.makeplots:

    profile_final = profile
    iters = np.arange(0, args.max_iterations)

    # save data final profile and flux
    profile_history[iter_idx, :] = profile
    flux_history[iter_idx, :] = flux

    # plot profile
    plt.figure()
    plt.plot(x, profile_final, 'b-', label='numerical solution')
    plt.plot(x, profile_ss, 'k--', label='analytic solution')
    plt.xlabel('x')
    plt.title('profile')
    plt.legend(loc='best')
    plt.grid()

    # plot residuals (excludes final value)
    plt.figure()
    plt.semilogy(iters, residual_history)
    plt.xlabel('iteration number')
    plt.title('Residual')
    plt.grid()

    # plot errors (excludes final value)
    plt.figure()
    plt.semilogy(iters, error_history)
    plt.xlabel('iteration number')
    plt.title('Max Error')
    plt.grid()

    # plot solution history (includes final value)
    fig, ax = plt.subplots()
    for i in range(args.plot_iters[0], args.plot_iters[1] + 1):
        plt.plot(x, profile_history[i,:], label=f'Iter {i}')
    plt.plot(x, profile_ss, 'k--', label='analytic solution')
    plt.xlabel('x')
    plt.title('Solution history')
    # place legend outside plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.grid()

    # plot flux history
    fig, ax = plt.subplots()
    for i in range(args.plot_iters[0], args.plot_iters[1] + 1):
        plt.plot(x, flux_history[i,:], label=f'Iter {i}')
    plt.plot(x, flux_ss, 'k--', label='analytic flux')
    plt.xlabel('x')
    plt.title('Flux history')
    # place legend outside plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.grid()

    # show plots
    plt.show()


# save the residual and error history
residual_error_history = np.vstack((residual_history, error_history))

outdir = "output"

if not os.path.exists(outdir):
    os.makedirs(outdir)

prefix = 'tango_pq'
if args.noise:
    prefix += 'noise'
prefix += '_p_' + str(args.p)
prefix += '_alpha_' + str(args.alpha)
prefix += '_beta_' + str(args.beta)

filename = prefix + '_residual_error_history.txt'

np.savetxt(os.path.join(outdir, filename), residual_error_history)
