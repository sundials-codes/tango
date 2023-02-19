#!/usr/bin/env python
"""
stripped_down_tango_example.py

Example for how to use tango to solve a turbulence and transport problem.

Here, the "turbulent flux" is specified analytically, using the example
(slightly modified and generalized) in the Shestakov et al. (2003) paper.
This example is a nonlinear diffusion equation with specified diffusion
coefficient and source.  There is a closed form answer for the steady
state solution which can be compared with the numerically found solution.
"""

# Adding the Tango directory to the PYTHONPATH environment variable is
# recommended. But if tango is not added to the PYTHONPATH, these commands can
# be used to add them manually.
# import sys
# sys.path.append("/path/to/tango")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from tango import derivatives
from tango import HToMatrixFD
from tango import lodestro_method

# slightly modified version of tango's noisyflux
from tango.extras import noisyflux_mod as noisyflux

import kinsol as kin

from autotune.search import *
from autotune.space import *
from autotune.problem import *

from computer import Computer
from options import Options
from data import Data
from gptune import GPTune
from database import GetMachineConfiguration

# ****** Input Options ****** #
def parse_args():

    import argparse

    parser = argparse.ArgumentParser(description='Run Shestakov example')

    # problem setup options
    parser.add_argument('--p', type=float, default=2.0,
                        help='power for analytic flux')

    parser.add_argument('--N', type=int, default=500,
                        help='number of spatial grid points')

    # initial guess options
    parser.add_argument('--IC', type=str, default='pow',
                        choices=['pow', 'const', 'lin', 'rand', 'solp', 'sol'],
                        help='set initial condition type')

    parser.add_argument('--IC_n0', type=float, default=2.0e-2,
                        help='boundary value at x = 0 (left) in pow IC')

    parser.add_argument('--IC_q', type=float, default=1.0,
                        help='power in pow IC')

    parser.add_argument('--IC_const', type=float, default=1.0,
                        help='value for constant IC')

    parser.add_argument('--IC_stddev', type=float, default=0.001,
                        help='standard deviation for rand IC')

    parser.add_argument('--IC_dev', type=float, default=0.1,
                        help='deviation for solp IC')

    # flux splitter options
    parser.add_argument('--Dmin', type=float, default=1e-5,
                        help='Minimum D value')

    parser.add_argument('--Dmax', type=float, default=1e13,
                        help='Maximum D value')

    parser.add_argument('--dpdxThreshold', type=float, default=10,
                        help='dpdx threshold value')

    # noisy flux options
    parser.add_argument('--addnoise', action='store_true',
                        help='add noise to flux values')

    parser.add_argument('--noise_Lac', type=float, default=0.2,
                        help='correlation length of noise')

    parser.add_argument('--noise_amplitude', type=float, default=0.1,
                        help='amplitude of noise')

    # KINSOL options
    parser.add_argument('--tol', type=float, default=1.0e-11,
                        help='Relaxation parameter for profile')

    parser.add_argument('--beta', type=float, default=1.0,
                        help='Relaxation parameter for profile')

    parser.add_argument('--beta_adapt', action='store_true',
                        help='Adapt relaxation (KINSOL)')

    parser.add_argument('--beta_adapt_factor', type=float, default=0.5,
                        help='Adapt relaxation factor (KINSOL)')

    parser.add_argument('--maxIters', type=int, default=500,
                        help='maximum number iterations')

    parser.add_argument('--mAA', type=int, default=0,
                        help='Anderson acceleration depth')

    parser.add_argument('--delayAA', type=int, default=0,
                        help='number of iterations to delay Anderson start')

    parser.add_argument('--adaptmAA', action='store_true',
                        help='adapt the acceleration depth')

    # output options
    parser.add_argument('--outputdir', type=str, default='output',
                        help='output directory')

    parser.add_argument('--gptune', action='store_true',
                        help='Run with GPTune')

    # parse command line args
    args = parser.parse_args()

    return args


# ****** Solution ****** #
def steady_state_solution(x, nL, p=2, S0=1, delta=0.1, L=1):
    """Return the exact steady state solution for the Shestakov test problem

    Inputs:
      x      Spatial coordinate grid (array)
      nL     boundary condition n(L) (scalar)
      p      parameter for power exponent in Shestakov diffusion (scalar)
      S0     parameter in source term --- amplitude (scalar)
      delta  parameter in source term --- location where it turns off (scalar)
      L      parameter for size of domain (scalar)
    Outputs:
    """
    a = 1 / (p+1)   # convenient shortcut

    nright = (nL**a + a * (S0 * delta)**a * (L-x))**(p+1)
    nleft = ((nL**a + a * (S0 * delta)**a *
              (L - delta + (p+1)/(p+2) * delta *
               (1 - (x/delta)**((p+2)/(p+1)))))**(p+1))
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
            dndx = derivatives.dx_centered_difference_edge_first_order(n,
                                                                       self.dx)
        else:
            dndx = derivatives.dx_centered_difference(n, self.dx)
        Deff = np.abs((dndx/n)**self.p)
        Gamma = -Deff * dndx
        return Gamma


# ****** Problem (Shestakov) ***** #
class Problem:

    def setup(args):

        # Domain length [0, L], number of grid points, and flux power
        L = 1.0
        N = args.N
        p = args.p

        # Mesh spacing and node locations
        Problem.dx = L / (N - 1)
        Problem.x = np.arange(N) * Problem.dx

        # Boundary condition at x = L
        Problem.nL = 1.0e-2

        # Time step size (1e4 is effectively infinite)
        Problem.dt = 1.0e4

        # Instantiate flux model
        if args.addnoise:
            fluxModel = FluxModel(Problem.dx, p=p)
            Problem.fluxModel = noisyflux.NoisyFlux(fluxModel,
                                                    args.noise_amplitude,
                                                    args.noise_Lac,
                                                    Problem.dx)
        else:
            Problem.fluxModel = FluxModel(Problem.dx, p=p)

        # Initialize FluxSplitter
        # for many problems, the exact value of these parameters doesn't matter
        # too much. These parameters have to do with the splitting between
        # diffusive and convective flux.
        thetaParams = {'Dmin': args.Dmin, 'Dmax': args.Dmax,
                       'dpdxThreshold': args.dpdxThreshold}

        Problem.fluxSplitter = lodestro_method.FluxSplit(thetaParams)

        # compute the analytic steady state solution
        Problem.nss = steady_state_solution(Problem.x, Problem.nL, p=p, L=L)

        # old time profile and initial guess
        n_IC = np.zeros_like(Problem.x)

        if args.IC == 'pow':
            # power law initial condition (q = 1 does not satisfy left BC)
            n0 = args.IC_n0
            q = args.IC_q
            n_IC[:] = ((Problem.nL - n0) / L**q) * Problem.x**q + n0
        elif args.IC == 'const':
            # constant initial condition
            n_IC[:] = args.IC_const
        elif args.IC == 'lin':
            # line between true solution at left and right boundary
            m = (Problem.nss[-1] - Problem.nss[0]) / L
            n_IC[:] = m * Problem.x + Problem.nss[0]
        elif args.IC == 'rand':
            # true solution plus random noise
            noise = np.random.normal(0, args.IC_stddev, Problem.nss.shape)
            n_IC[:] = Problem.nss[:] + noise[:]
        elif args.IC == 'solp':
            # true solution plus fixed perturbation
            n_IC[:] = Problem.nss[:] + args.IC_dev * Problem.nss[:]
        else:
            # true solution
            n_IC[:] = Problem.nss[:]

        # enforce right BC for all inital conditions
        n_IC[-1] = Problem.nL

        # set for old time
        Problem.n_mminus1 = np.copy(n_IC)

        # print problem setup to screen
        print("Tango Shestakov Example:")
        print("  Domain size L            =", L)
        print("  Mesh points N            =", N)
        print("  Mesh spacing dx          =", Problem.dx)
        print("  Right boundary value     =", Problem.nL)
        print("  Time step size           =", Problem.dt)
        print("  D minimum                =", args.Dmin)
        print("  D maximum                =", args.Dmax)
        print("  dp/dx threshold          =", args.dpdxThreshold)
        print("  Flux power               =", p)
        print("  Relaxation beta          =", args.beta)
        print("  Max iterations           =", args.maxIters)
        print("  Acceleration depth       =", args.mAA)
        print("  Acceleration delay       =", args.delayAA)
        print("  Adapt Acceleration depth =", args.adaptmAA)
        print("  Initial condition        =", args.IC)
        if args.IC == 'pow':
            print("  IC power                 =", args.IC_q)
            print("  IC value left boundary   =", args.IC_n0)
        elif args.IC == 'const':
            print("  IC const                 =", args.IC_const)
        elif args.IC == 'rand':
            print("  IC stddev                =", args.IC_stddev)
        elif args.IC == 'solp':
            print("  IC dev                   =", args.IC_dev)

        # always save residual history
        Problem.F_hist = np.zeros((args.maxIters, N))
        Problem.R_hist = np.zeros((args.maxIters, N))


    def Gfun(sunvec_profile_old, sunvec_profile_new, user_data):

        # extract arrays
        profile_old = kin.N_VGetData(sunvec_profile_old)
        profile_new = kin.N_VGetData(sunvec_profile_new)

        # get turbulent flux
        try:
            flux = Problem.fluxModel.get_flux(profile_old)
        except Exception as e:
            print(e)
            print("\nERROR: get_flux failed")
            return -1

        # transform flux into effective transport coefficients.  H2=D, H3=-c
        # [use flux split class from lodestro_method]
        try:
            (D, c, _) = Problem.fluxSplitter.flux_to_transport_coeffs(flux,
                                                                      profile_old,
                                                                      Problem.dx)
        except Exception as e:
            print(e)
            print("\nERROR: flux_to_transport_coeffs failed")
            return -1

        # H's represent terms in the transport equation
        # H2const could represent a background level of (classical) diffusion
        H1 = np.ones_like(Problem.x)
        H7 = source(Problem.x)
        H2Turb = D
        H2const = 0.00
        H2 = H2Turb + H2const
        H3 = -c

        # construct equation for the new profile
        try:
            (A, B, C, f) = HToMatrixFD.H_to_matrix(Problem.dt, Problem.dx,
                                                   Problem.nL, Problem.n_mminus1,
                                                   H1, H2=H2, H3=H3, H7=H7)
        except Exception as e:
            print(e)
            print("\nERROR: matrix setup failed")
            return -1

        # see fieldgroups.calculate_residual() for additional information on
        # the residual calculation
        resid = (A * np.concatenate((profile_old[1:], np.zeros(1))) +
                 B * profile_old +
                 C * np.concatenate((np.zeros(1), profile_old[:-1]))
                 - f)

        # compute normalized residual
        resid = resid / np.max(np.abs(f))

        # solve matrix equation for new profile n_{i+1} = G(n_i)
        try:
            profile_new[:] = HToMatrixFD.solve(A, B, C, f)
        except Exception as e:
            print(e)
            print("\nERROR: matrix solve failed")
            return -1

        # compute F_i = G(n_i) - n_i (same as in KINSOL)
        Problem.F_hist[Problem.numGEvals, :] = profile_new - profile_old
        Problem.R_hist[Problem.numGEvals, :] = resid

        # update number of G evals
        Problem.numGEvals += 1

        # update iteration count
        Problem.numIters += 1

        return 0


    def solveKINSOL(**kwargs):

        # Reset counts and saved values
        Problem.numGEvals = 0
        Problem.numIters = 0
        Problem.F_hist[:] = float("nan")
        Problem.R_hist[:] = float("nan")

        # solution and scaling arrays
        profile_new = np.copy(Problem.n_mminus1)
        scale = np.ones_like(Problem.n_mminus1)

        # create N_Vector objects
        sunvec_profile = kin.N_VMake_Serial(profile_new)
        sunvec_scale = kin.N_VMake_Serial(scale)

        # allocate memory for KINSOL
        kmem = kin.KINCreate()

        # set number of prior residuals used in Anderson acceleration
        if "mAA" in kwargs:
            flag = kin.KINSetMAA(kmem, kwargs["mAA"])
            if flag < 0:
                raise RuntimeError(f'KINSetMAA returned {flag}')

            if "adaptmAA" in kwargs:
                flag = kin.KINSetAdaptiveMAA(kmem, int(kwargs["adaptmAA"]))
                if flag < 0:
                    raise RuntimeError(f'KINSetAdaptiveMAA returned {flag}')

        # wrap the python system function so that it is callable from C
        sysfn = kin.WrapPythonSysFn(Problem.Gfun)

        # initialize KINSOL
        flag = kin.KINInitPy(kmem, sysfn, sunvec_profile)
        if flag < 0:
            raise RuntimeError(f'KINInitPy returned {flag}')

        # specify stopping tolerance based on residual
        if "tol" in kwargs:
            flag = kin.KINSetFuncNormTol(kmem, kwargs["tol"])
            if flag < 0:
                raise RuntimeError(f'KINSetFuncNormTol returned {flag}')

        # ignore convergence test and run to max iterations
        if "maxIters" in kwargs:
            flag = kin.KINSetNumMaxIters(kmem, kwargs["maxIters"])
            if flag < 0:
                raise RuntimeError(f'KINSetSetNumMaxIters returned {flag}')

        # ignore convergence test and run to max iterations
        # flag = kin.KINSetUseMaxIters(kmem, 1)
        # if flag < 0:
        #     raise RuntimeError(f'KINSetUseMaxIters returned {flag}')

        # return the newest iteration at end
        flag = kin.KINSetReturnNewest(kmem, 1)
        if flag < 0:
            raise RuntimeError(f'KINSetReturnNewest returned {flag}')

        # set Anderson acceleration delay
        if "delayAA" in kwargs:
            flag = kin.KINSetDelayAA(kmem, kwargs["delayAA"])
            if flag < 0:
                raise RuntimeError(f'KINSetDelayAA returned {flag}')

        # set fixed point and Anderson acceleration damping
        if "beta" in kwargs:
            flag = kin.KINSetDamping(kmem, kwargs["beta"])
            if flag < 0:
                raise RuntimeError(f'KINSetDamping returned {flag}')

            flag = kin.KINSetDampingAA(kmem, kwargs["beta"])
            if flag < 0:
                raise RuntimeError(f'KINSetDampingAA returned {flag}')

        if "beta_adapt" in kwargs:
            flag = kin.KINSetAdaptiveDampingAA(kmem, int(kwargs["beta_adapt"]))
            if flag < 0:
                raise RuntimeError(f'KINSetAdaptiveDampingAA returned {flag}')

        if "beta_adapt_factor" in kwargs:
            flag = kin.KINSetAdaptiveDampingFactorAA(kmem,
                                                     kwargs["beta_adapt_factor"])
            if flag < 0:
                raise RuntimeError(f'KINSetAdaptiveDampingFactorAA returned {flag}')

        # set error log file
        flag = kin.KINSetErrFilename(kmem, "kinsol_error.log")
        if flag < 0:
            raise RuntimeError(f'KINSetErrFilename returned {flag}')

        # set info file
        flag = kin.KINSetInfoFilename(kmem, "kinsol_info.log")
        if flag < 0:
            raise RuntimeError(f'KINSetInfoFilename returned {flag}')

        # set info print level
        flag = kin.KINSetPrintLevel(kmem, 0)
        if flag < 0:
            raise RuntimeError(f'KINSetPrintLevel returned {flag}')

        # Call KINSOL to solve problem
        kin_flag = kin.KINSol(kmem,            # KINSOL memory block
                              sunvec_profile,  # initial guess; solution vector
                              kin.KIN_FP,      # global strategy choice
                              sunvec_scale,    # scaling vector for the variable
                              sunvec_scale)    # scaling vector for function values

        # Print solution and solver statistics
        flag, fnorm = kin.KINGetFuncNorm(kmem)
        if flag < 0:
            raise RuntimeError(f'KINGetFuncNorm returned {flag}')

        print('Computed solution (||F|| = %Lg):' % fnorm)
        print('Interations:', Problem.numIters)

        if kin_flag < 0:
            print(f'KINSol failed with return value {kin_flag}')
            return 200
        elif kin_flag > 0:
            print(f'KINSol returned {kin_flag}')
        else:
            print('KINSol finished')

        # Free memory
        # kin.KINFree(kmem)
        # kin.N_VDestroy(sunvec_profile)
        # kin.N_VDestroy(sunvec_scale)

        return Problem.numIters


def objectives(point):

    print(point)
    iters = Problem.solveKINSOL(**point)
    return [iters]

def runGPTune(args):

    import os
    global nodes
    global cores

    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    input_space = Space([Integer(2, 50, name="p")])
    output_space = Space([Real(0, float('Inf'), name="iters", optimize=True)])

    parameters = ["beta"]
    parameters_list = list()
    if "beta" in parameters:
        parameters_list.append(Real(0.0, 1.0, name="beta"))
    parameter_space = Space(parameters_list)

    constraints = dict()
    if "beta" in parameters:
        constraints["cst1"] = "beta > 0.0 and beta < 1.0"

    constants = dict(vars(args))
    matches = list()
    del constants['p']
    for key in parameters:
        if key in constants:
            matches.append(key)
    for m in matches:
        del constants[m]
    for key, value in constants.items():
        if isinstance(value, bool):
            print(f"{key} is {value}")
            constants[key] = int(value)
    print(constants)

    problem = TuningProblem(input_space, parameter_space, output_space,
                            objectives, constraints, constants=constants)
    computer = Computer(nodes=nodes, cores=cores, hosts=None)
    options = Options()
    options["lite_mode"] = True
    options['verbose'] = False
    options.validate(computer=computer)

    data = Data(problem)
    gptune = GPTune(problem, computer=computer, data=data, options=options,
                    driverabspath=os.path.abspath(__file__))

    giventask = [[args.p]]
    (data, models, stats) = gptune.SLA(20, 10, Tgiven=giventask)

    print("stats: ", stats)
    """ Print all input and parameter samples """
    print("  Tasks:", data.I)
    print("  Parameter Samples:", data.P)
    print("  Outputs:", data.O)
    print(f"    Output[{np.argmin(data.O)}] = {data.O[np.argmin(data.O)]}")
    print(f"    Params[{np.argmin(data.O)}] = {data.P[np.argmin(data.O)]}")


# ****** Main ***** #
def main():

    import os

    args = parse_args()

    # setup the problem
    Problem.setup(args)

    if args.gptune:
        print("Running GPUTune")
        runGPTune(args)
    else:
        # solve the problem
        Problem.solveKINSOL(**vars(args))

        # print final resiudal and error
        print("Finished:")
        print("  Interations =", Problem.numIters)

        # iteration range to plot

        # initial to end - 1 (length numIters)
        iters = np.arange(0, Problem.numIters)

        # initial to end (length numIters + 1)
        itersp1 = np.arange(0, Problem.numIters + 1)

        # write history to file

        outdir = args.outputdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # add a prefix for different configurations
        prefix = 'p_' + str(args.p)
        prefix = prefix + '_beta_' + str(args.beta)
        prefix = prefix + '_adapt-beta_' + str(args.beta_adapt)
        prefix = prefix + '_adapt-beta-factor_' + str(args.beta_adapt_factor)
        prefix = prefix + '_m_' + str(args.mAA)
        prefix = prefix + '_delay_' + str(args.delayAA)
        prefix = prefix + '_adapt-m_' + str(args.adaptmAA)
        if args.addnoise:
            prefix = prefix + '_noise'

        # save residual norm history
        resF_nrm = np.zeros((Problem.numIters, 1))
        resR_nrm = np.zeros((Problem.numIters, 1))
        for i in iters:
            resF_nrm[i] = np.sqrt(np.sum(Problem.F_hist[i, :]**2))
            resR_nrm[i] = np.sqrt(np.sum(Problem.R_hist[i, :]**2))
            # resF_nrm[i] = np.sqrt(np.mean(Problem.F_hist[i, :]**2))
            # resR_nrm[i] = np.sqrt(np.mean(Problem.R_hist[i, :]**2))
        np.savetxt(outdir + '/' + prefix + '_Fresid.txt', resF_nrm)
        np.savetxt(outdir + '/' + prefix + '_Rresid.txt', resR_nrm)

# ****** run main ****** #
if __name__ == '__main__':
    import sys
    sys.exit(main())
