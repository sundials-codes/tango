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

    import argparse

    parser = argparse.ArgumentParser(description='Run Shestakov example')

    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Relaxation parameter')
    parser.add_argument('--p', type=float, default=2.0,
                        help='power for analytic flux')
    parser.add_argument('--L', type=float, default=1.0,
                        help='domain size [0,L]')
    parser.add_argument('--N', type=int, default=500,
                        help='number of spatial grid points')
    parser.add_argument('--nL', type=float, default=1e-2,
                        help='right boundary value')
    parser.add_argument('--dt', type=float, default=1e4,
                        help='time step size')
    parser.add_argument('--maxiters', type=int, default=200,
                        help='maximum number iterations')
    parser.add_argument('--Dmin', type=float, default=1e-5,
                        help='Minimum D value')
    parser.add_argument('--Dmax', type=float, default=1e13,
                        help='Maximum D value')
    parser.add_argument('--dpdxThreshold', type=float, default=10,
                        help='dpdx threshold value')

    # parse command line args
    args = parser.parse_args()

    #### create stuff
    maxIterations = args.maxiters
    alpha = args.alpha  # relaxation parameter on the effective diffusion coefficient
    p = args.p          # power for analytic flux

    # problem setup
    L = args.L          # size of domain
    N = args.N          # number of spatial grid points
    dx = L / (N-1)      # spatial grid size
    x = np.arange(N)*dx # location corresponding to grid points j=0, ..., N-1

    # initial condition
    n_IC = 0.02 - 0.01 * x
    #n_IC = nSave + .0001
    n_mminus1 = n_IC
    profile = n_IC

    # boundary condition
    nL = args.nL

    # time step  (effectively infinite)
    dt = args.dt

    # instantiate flux model
    fluxModel = FluxModel(dx, p=p)

    # initialize data storage
    nAll = np.zeros((maxIterations, N))
    fluxAll = np.zeros_like(nAll)
    residualHistory = np.zeros(maxIterations)

    # initialize FluxSplitter.
    # for many problems, the exact value of these parameters doesn't matter too much.
    #  these parameters have to do with the splitting between diffusive and convective flux.
    thetaParams = {'Dmin': args.Dmin, 'Dmax': args.Dmax, 'dpdxThreshold': args.dpdxThreshold}

    fluxSplitter = lodestro_method.FluxSplit(thetaParams)


    for iterationNumber in np.arange(0, maxIterations):
        # get next turbulent flux
        flux = fluxModel.get_flux(profile)

        # (put any inner iteration loop here)

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
        resid = resid / np.max(np.abs(f))  # normalize the residual
        rmsResid = np.sqrt( np.mean( resid**2 ))  # take an rms characterization of residual
        residualHistory[iterationNumber] = rmsResid

        # solve matrix equation for new profile
        profile = HToMatrixFD.solve(A, B, C, f)

        # save data
        nAll[iterationNumber, :] = profile
        fluxAll[iterationNumber, :] = flux

        # check
        if np.any(profile < 0) == True:
            print(f'error.  negative value detected in profile at l={iterationNumber}')
            break


    # finish

    nFinal = profile
    iters = np.arange(0, maxIterations)


    # plot the last iteration of density --- presumably the correct solution, if converged
    #  Also plot the analytic steady state solution
    nss = steady_state_solution(x, nL, p=p)

    plt.figure()
    plt.plot(x, nFinal, 'b-', label='numerical solution')
    plt.plot(x, nss, 'k--', label='analytic solution')
    plt.xlabel('x')
    plt.ylabel('n')
    plt.title('Final Solution')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    # plot residual
    plt.figure()
    plt.semilogy(iters, residualHistory)
    plt.xlabel('iteration number')
    plt.ylabel('Normalized RMS Norm of Residual')
    plt.title('Final Residual')
    plt.grid()
    plt.show()



    # plots of first few iterations if desired
    #plt.figure()
    #plt.plot(x, nAll[0], 'k--')
    #plt.plot(x, nAll[1], 'r--')
    #plt.plot(x, nAll[2], 'b--')
    #plt.plot(x, nAll[3], 'k-')
    #plt.plot(x, nAll[4], 'r-')
    #plt.plot(x, nAll[5], 'b-')
    #plt.xlabel('x')
    #plt.title('n')


    #plt.figure()
    #plt.plot(x, fluxAll[0], 'k--')
    #plt.plot(x, fluxAll[1], 'r--')
    #plt.plot(x, fluxAll[2], 'b--')
    #plt.plot(x, fluxAll[3], 'k-')
    #plt.plot(x, fluxAll[4], 'r-')
    #plt.plot(x, fluxAll[5], 'b-')
    #plt.xlabel('x')
    #plt.title('flux')

    #nSave = nFinal


# ****** run main ****** #
if __name__ == '__main__':
    import sys
    sys.exit(main())
