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

import sys

import numpy as np
import matplotlib.pyplot as plt
import kinsol as kin

#from tango.extras import noisyflux
import modified_noisyflux as noisyflux
from tango import derivatives
from tango import HToMatrixFD
from tango import lodestro_method

from matplotlib.animation import FuncAnimation, PillowWriter

import warnings
#warnings.filterwarnings('error')

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

class Problem:
    #### create stuff
    p = 2     # power for analytic flux
    alpha = 0.1
    depth = 0
    beta = 1.0

    maxIterations = 300
    np.set_printoptions(precision=24)
    tol = 1e-8
    noise_Lac = 0.2  # correlation length of noise
    noise_amplitude = 0

    iterationNumber = 0

    # problem setup
    L = 1           # size of domain
    N = 500          # number of spatial grid points
    dx = L / (N-1)  # spatial grid size
    x = np.arange(N)*dx # location corresponding to grid points j=0, ..., N-1

    # initial condition
    #n_IC = 0.02 - 0.01 * x
    n_IC = 1.0 + 0.0 * x
    #n_IC = nSave + .0001
    n_mminus1 = np.zeros_like(n_IC)
    n_mminus1[:] = n_IC
    profile = n_IC

    # boundary condition
    nL = 1e-2
    #nL = 0.0

    # time step  (effectively infinite)
    dt = 1e-5
    #dt = 1e-2

    # instantiate flux model
    fluxModel = FluxModel(dx, p=p)

    # Add decorator to apply noise first, then apply buffer
    #fluxModel = noisyflux.NoisyFlux(fluxModel, noise_amplitude, noise_Lac, dx)   # for AR(1) noise

    # initialize data storage
    nAll = np.zeros((maxIterations, N))
    fluxAll = np.zeros_like(nAll)
    residualHistory = np.zeros(maxIterations)
    myResidHistory = np.zeros(maxIterations)

    # initialize FluxSplitter.  
    # for many problems, the exact value of these parameters doesn't matter too much.
    #  these parameters have to do with the splitting between diffusive and convective flux.
    thetaParams = {'Dmin': 1e-5, 'Dmax': 1e13, 'dpdxThreshold': 10} 
       
    fluxSplitter = lodestro_method.FluxSplit(thetaParams)

    D_EWMA = 0
    c_EWMA = 0

    resid = 0

    def G(sunvec_n, sunvec_g, user_data):
        profile = kin.N_VGetData(sunvec_n)
        g = kin.N_VGetData(sunvec_g)
        
        # get next turbulent flux
        flux = Problem.fluxModel.get_flux(profile)

        # (put any inner iteration loop here)

        # transform flux into effective transport coefficients.  H2=D, H3=-c
        # [use flux split class from lodestro_method]
        (D, c, _) = Problem.fluxSplitter.flux_to_transport_coeffs(flux, 
                                                                profile,
                                                                Problem.dx)

        # compute relaxation of D, c  (EWMA = Exponentially Weighted Moving Average)
        if Problem.iterationNumber == 0:
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
        (A, B, C, f) = HToMatrixFD.H_to_matrix(Problem.dt, 
                                            Problem.dx, 
                                            Problem.nL, 
                                            Problem.n_mminus1, H1, H2=H2, H3=H3, H7=H7)

        # see fieldgroups.calculate_residual() for additional information on the residual calculation
        resid = A*np.concatenate((profile[1:], np.zeros(1))) + B*profile + C*np.concatenate((np.zeros(1), profile[:-1])) - f
        resid = resid / np.max(np.abs(f))  # normalize the residual
        rmsResid = np.sqrt( np.mean( resid**2 ))  # take an rms characterization of residual
        Problem.residualHistory[Problem.iterationNumber] = rmsResid

        # solve matrix equation for new profile
        g[:] = HToMatrixFD.solve(A, B, C, f)

        # save data
        Problem.nAll[Problem.iterationNumber, :] = g
        Problem.fluxAll[Problem.iterationNumber, :] = flux

        diff = g - profile
        Problem.myResidHistory[Problem.iterationNumber] = np.amax(np.abs(diff))

        # has to increment interal iteration count
        Problem.iterationNumber += 1

        return 0

### MAIN
if "beta" in sys.argv:
    Problem.beta = float(sys.argv[sys.argv.index("beta")+1])

if "depth" in sys.argv:
    Problem.depth = int(sys.argv[sys.argv.index("depth")+1])

if "c" in sys.argv:
    Problem.c = float(sys.argv[sys.argv.index("c")+1])
    Problem.alpha = Problem.c / Problem.p

if "alpha" in sys.argv:
    Problem.alpha = float(sys.argv[sys.argv.index("alpha")+1])

if "p" in sys.argv:
    Problem.fluxModel.p = int(sys.argv[sys.argv.index("p")+1])

if "maxIterations" in sys.argv:
    Problem.maxIterations = int(sys.argv[sys.argv.index("maxIterations")+1])

# vector
sunvec_n = kin.N_VMake_Serial(Problem.n_IC)
scale = np.zeros(Problem.N)
sunvec_scale = kin.N_VMake_Serial(scale)

# memory
kmem = kin.KINCreate()
flag = kin.KINSetMAA(kmem, Problem.depth)
flag = kin.KINSetNumMaxIters(kmem, Problem.maxIterations)
sysfn = kin.WrapPythonSysFn(Problem.G)
flag = kin.KINInitPy(kmem, sysfn, sunvec_n)

# options
flag = kin.KINSetFuncNormTol(kmem, Problem.tol)
flag = kin.KINSetErrFilename(kmem, "error.log")
flag = kin.KINSetDampingAA(kmem, Problem.beta)
#flag = kin.KINSetInfoFile(kmem, "kinfo.log")

# solve
kin.N_VConst(1.0, sunvec_scale)
try:
    flag = kin.KINSol(kmem,
                      sunvec_n,
                      kin.KIN_FP,
                      sunvec_scale,
                      sunvec_scale)
except:
    pass

# finish

nFinal = Problem.profile
#print(nFinal)
iters = np.arange(0,Problem.maxIterations)
#convIter = np.argmax(Problem.residualHistory < Problem.tol)
#print(convIter)
print(Problem.iterationNumber)

# plot the last iteration of density --- presumably the correct solution, if converged
#  Also plot the analytic steady state solution
nss = steady_state_solution(Problem.x, Problem.nL, p=Problem.p)
#print(nss - Problem.g)
print( np.amax(np.abs(nFinal - nss)) )

if 'plot' in sys.argv:
    print('Plotting...')
    fig, ax = plt.subplots()
    x = Problem.x
    y = []
    ln1, = plt.plot(x,nss,'k--', label='analytic solution')
    ln2, = plt.plot([],[],'b-', label='numerical solution')

    def init():
        ax.set_xlim(0,Problem.L)
        ax.set_ylim(0,0.05)

    def update(i):
        y = Problem.nAll[i,:]
        ax.set_ylim(0,np.amax(y)+0.01)
        ln2.set_data(x,y)
        if i+1 == Problem.iterationNumber:
            ln2.set_color('red')

    ani = FuncAnimation(fig, update, Problem.iterationNumber, init_func=init)
    writer = PillowWriter(fps=1)
    ani.save('solution.gif', writer=writer)

    # plot residuals
    plt.figure()
    plt.semilogy(iters, Problem.residualHistory)
    plt.xlabel('iteration number')
    plt.title('RMS residual')
    plt.savefig('rms.png')

    plt.figure()
    plt.semilogy(iters, Problem.myResidHistory)
    plt.xlabel('iteration number')
    plt.title('Max-norm Residual')
    plt.savefig('residual.png')

if 'residplot' in sys.argv:
    print('Plotting residual...')
    fig, ax = plt.subplots()
    x = Problem.x
    y = []
    ln2, = plt.plot([],[],'b-', label='residual')

    def init():
        ax.set_xlim(0,Problem.L)
        #ax.set_ylim(-0.01,0.01)
        ax.set_yscale('symlog', linthreshy=1e-8)

    def update(i):
        y1 = Problem.nAll[i+1,:]
        y2 = Problem.nAll[i,:]
        y = y1 - y2
        ln2.set_data(x,y)
        if i == Problem.iterationNumber:
            ln2.set_color('red')

    ani = FuncAnimation(fig, update, Problem.iterationNumber-1, init_func=init)
    writer = PillowWriter(fps=5)
    ani.save('residual.gif', writer=writer)

if 'save' in sys.argv:
    print('Saving residual data...')
    fname = sys.argv[sys.argv.index("save")+1]
    np.savetxt(fname + "resid", Problem.myResidHistory[0:Problem.iterationNumber])
    np.savetxt(fname + "iters", iters)
