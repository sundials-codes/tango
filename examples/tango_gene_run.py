from __future__ import division, absolute_import
import numpy as np


# Tango imports... should simplify these so I'm not importing them all separately!
import tango
import tango.gene_startup
import tango.smoother
import tango.genecomm_unitconversion
import tango.logging
from tango.extras import util # for duration_as_hms

def initialize_iteration_parameters():
    maxIterations = 3
    thetaParams = {'Dmin': 1e-5,
                   'Dmax': 1e3,
                   'dpdxThreshold': 400000}
    EWMAParamTurbFlux = 0.1
    EWMAParamProfile = 0.1
    lmParams = {'EWMAParamTurbFlux': EWMAParamTurbFlux,
            'EWMAParamProfile': EWMAParamProfile,
            'thetaParams': thetaParams}
    tol = 1e-11  # tol for convergence... reached when a certain error < tol
    return (maxIterations, lmParams, tol)

def temperature_initial_condition(rho):
    """Initial temperature profile
    Inputs:
      rho       normalized radial coordinate rho=r/a (array)
    Outputs:
      T         temperature profile in SI (array)
    """
    e = 1.60217662e-19          # electron charge
    
    kappa_T = 6.96
    deltar = 0.9
    rho0 = 0.5
    rhominus = rho - rho0 + deltar/2
    deltaT = 0.1

    e = 1.60217662e-19
    T0 = 1000*e
    invasp = 0.36
    T = T0 * np.exp( -kappa_T * invasp * (rho - rho0 - deltaT * (np.tanh(rhominus/deltaT) - np.tanh(deltar/2/deltaT))));
    ind = int(np.abs(rho - (rho0 - deltar/2)).argmin())
    ind2 = (rho < (rho0-deltar/2));
    T[ind2] = T[ind];
    return T
    
    
def source_fhat(rho):
    """Provide the function fhat in the source term V'S = A fhat(rho)
    
    Inputs:
      rho           radial coordinate rho=r/a (array)
    Outputs:
      fhat          output (array)
    """
    rho_a = 0.2
    rho_b = 0.4
    rho0 = 0.3
    w = 0.05
    
    fhat = np.zeros_like(rho)
    ind = (rho > rho_a) & (rho < rho_b)
    fhat[ind] = np.exp( -(rho[ind] - rho0)**2 / w**2)
    return fhat

def source_H7(r, minorRadius, majorRadius, A):
    """Provide the source term V'S to the transport equation.  The contribution to H7 is V'S.
    
    The source is written as V'S = A * f(r) = A * fhat(r/a)
    
    Inputs:
      r             radial coordinate r, in m (array)
      minorRadius   minor radius a, in m (scalar)
      majorRadius   major radius R0, in m (scalar)
      A             amplitude in SI of V'S, in SI (scalar)
    Outputs:
      H7contrib     contribution to H7 (=V'S) (array)
    """
    rho = r / minorRadius
    fhat = source_fhat(rho)
    H7contrib = A * fhat
    return H7contrib

def add_one_if_even(n):
    """if n is an even integer, add one and return an odd integer.  If n is an odd integer, return n"""
    if n % 2 == 0:
        return n + 1
    elif n % 2 == 1:
        return n
    else:
        raise ValueError('n does not appear to be an integer.')
    
class ComputeAllH(object):
    def __init__(self, turbhandler, Vprime, minorRadius, majorRadius, A):
        self.turbhandler = turbhandler
        self.Vprime = Vprime
        self.minorRadius = minorRadius
        self.majorRadius = majorRadius
        self.A = A
    def __call__(self, t, r, pressure):
        """Define the contributions to the H coefficients
        
        Inputs:
          t         time (scalar)
          r       radial coordinate in SI (array)
          pressure  pressure profile in SI (array)
        """
        H1 = 1.5 * self.Vprime
        
        # turbulent flux
        (H2_turb, H3_turb, extradata) = self.turbhandler.Hcontrib_turbulent_flux(pressure)
        
        # add some manually input diffusivity
        D_adhoc = 0.05  # in SI, m^2/s
        H2_adhoc = D_adhoc * self.Vprime

        # sum up the turbulent + adhoc transport terms
        H2 = H2_turb + H2_adhoc
        H3 = H3_turb
        
        H4 = None
        H6 = None
        
        # Heat Source
        H7 = source_H7(r, self.minorRadius, self.majorRadius, self.A)
        
        return (H1, H2, H3, H4, H6, H7, extradata)
        
def problem_setup():
    """
    Still to be done: need to choose good profiles for
        ion temperature initial condition
        plasma density
        heat source
        tango and GENE radial grids
    """
    # Set up problem parameters
    minorRadius = 0.594     # minor radius a, in meters
    majorRadius = 1.65      # major radius R0, in meters
    
    # Set up the turbulence (GENE) grid
    rhoLeftBndyGene = 0.1
    rhoRightBndy = 0.9
    numRadialPtsGene = 120  # from parameters file... needs to be consistent
    rhoGene = np.linspace(rhoLeftBndyGene, rhoRightBndy, numRadialPtsGene)
    rGene = rhoGene * minorRadius
    drGene = rGene[1] - rGene[0]
    
    # Set up the transport (Tango) grid
    numRadialPtsTango = 27
    drho = rhoRightBndy / (numRadialPtsTango - 0.5)  # spatial grid size
    rhoTango = np.linspace(drho/2, rhoRightBndy, numRadialPtsTango)  # Tango inner-most point is set at delta rho/2, not exactly zero.
    rTango = rhoTango * minorRadius # physical radius r, measured in meters, used as the independent coordinate in Tango
    drTango = rTango[1] - rTango[0]
    L = rTango[-1]  # size of domain
    
    VprimeGene = 4 * np.pi**2 * majorRadius * rGene
    VprimeTango = 4 * np.pi**2 * majorRadius * rTango
    gradPsiSqTango = np.ones_like(rTango) # |grad r|^2 = 1
    
    Bref = 1.14
    B0 = Bref
    Lref = 1.65
    Tref = 1
    nref = 3.3
    ionMass = 2  # in proton masses
    mref = ionMass
    ionCharge = 1
    #mref = 2  # do I need mref??
    
    
    rho0 = 0.5
    kappa_n = 2.22;
    invasp = minorRadius / majorRadius
    densityProfileGene = 3.3e19 * np.exp(-kappa_n * invasp * (rhoGene - rho0))
    densityProfileTango = 3.3e19 * np.exp(-kappa_n * invasp * (rhoTango - rho0))
    
    # create object for interfacing tango and GENE radial grids
       # must be consistent with whether Tango's or Gene's radial domain extends farther radially outward
    rExtrapZoneLeft = 0.75 * minorRadius
    rExtrapZoneRight = 0.81 * minorRadius
    polynomialDegree = 1
    gridMapper = tango.interfacegrids_gene.TangoOutsideExtrapCoeffs(rTango, rGene, rExtrapZoneLeft, rExtrapZoneRight, polynomialDegree)
    
    
    # specify a boundary condition for pressure at the outward radial boundary
    temperatureRightBCInkeV = 0.44
    e = 1.60217662e-19          # electron charge
    temperatureRightBC = temperatureRightBCInkeV * 1000 * e  # temperature in SI
    pressureRightBC = temperatureRightBC * densityProfileTango[-1]
    
    # specify a temperature and pressure initial condition
    temperatureICTango = temperature_initial_condition(rhoTango) # in SI
    pressureICTango = temperatureICTango * densityProfileTango
    
    # specify safety factor
    safetyFactorGeneGrid = 0.868 + 2.2 * rhoGene**2
    
    # GENE setup
    fromCheckpoint = True    # true if restarting a simulation from an already-saved checkpoint
    (geneFluxModel, MPIrank) = tango.gene_startup.setup_gene_run(rTango, rGene, minorRadius, majorRadius, B0, ionMass, ionCharge, densityProfileTango, pressureICTango, safetyFactorGeneGrid,
                                                                 Bref, Lref, Tref, nref, gridMapper, fromCheckpoint)
    
    
    # other transport physics / physicsToH object creation
    #profilesAll = tango.physics.initialize_profile_defaults(ionMass, densityProfile, psiTango, minorRadius, majorRadius, B0, Vprime, gradPsiSq)
    #HcontribTransportPhysics = tango.physics_to_H.Hcontrib_TransportPhysics(profilesAll)
    
    # iteration parameters setup
    (maxIterations, lmParams, tol) = initialize_iteration_parameters()    
    
    # create flux smoother for spatial averaging of flux
    windowSizeInGyroradii = 10
    rhoref = tango.genecomm_unitconversion.rho_ref(Tref, mref, Bref)
    windowSizeInPoints = int( np.round(windowSizeInGyroradii * rhoref / drGene) )
    windowSizeInPoints = add_one_if_even(windowSizeInPoints)  # ensure windowSizeInPoints is odd for Smoother
    fluxSmoother = tango.smoother.Smoother(windowSizeInPoints)
    
    # creation of turbulence handler
    turbhandler = tango.lodestro_method.TurbulenceHandler(drGene, rTango, lmParams, geneFluxModel, gridMapper=gridMapper, VprimeTango=VprimeTango, fluxSmoother=fluxSmoother)
    
    # specify a source function amplitude
    A = 57.3e6 # in SI, W/m^3  # for total input power of 3 MW
    
    # initialize the compute all H object
    compute_all_H = ComputeAllH(turbhandler, VprimeTango, minorRadius, majorRadius, A)
    t_array = np.array([0, 1e4])  # specify the timesteps to be used.
    return (MPIrank, L, rTango, pressureRightBC, pressureICTango, maxIterations, tol, geneFluxModel, turbhandler, compute_all_H, t_array)

    
# ************************************************** #
####              START OF MAIN PROGRAM           ####
# ************************************************** #

(MPIrank, L, rTango, pressureRightBC, pressureICTango, maxIterations, tol, geneFluxModel, turbhandler, compute_all_H, t_array) = problem_setup()



# set up FileHandlers
f1HistoryHandler = tango.handlers.Savef1HistoryHandler(iterationInterval=1, basename='f1_iteration_history', genefile='checkpoint_000')
tangoCheckpointHandler = tango.handlers.TangoCheckpointHandler(iterationInterval=1, basename='tango_checkpoint')
tangoHistoryHandler = tango.handlers.TangoHistoryHandler(iterationInterval=1, basename='tango_history', maxIterations=maxIterations)

# specify how long GENE runs between Tango iterations.  Specified in Lref/cref
geneFluxModel.set_simulation_time(50)

solver = tango.solver.Solver(L, rTango, pressureICTango, pressureRightBC, t_array, maxIterations, tol, compute_all_H, turbhandler)

## Set up the file handling 
parallelEnvironment = True
solver.fileHandlerExecutor.set_parallel_environment(parallelEnvironment, MPIrank)
solver.fileHandlerExecutor.add_handler(f1HistoryHandler)
solver.fileHandlerExecutor.add_handler(tangoCheckpointHandler)
solver.fileHandlerExecutor.add_handler(tangoHistoryHandler)

# create parameters for dataSaverHandler
arraysToSave = ['H2', 'H3', 'profile', 
                'D', 'c', 
                'profileEWMATurbGrid',
                'fluxTurbGrid', 'fluxEWMATurbGrid',
                'DHatTurbGrid', 'cHatTurbGrid', 'thetaTurbGrid']  # for list of possible arrays, see solver._pkgdata()
databasename = 'datasaver'
solver.dataSaverHandler.initialize_datasaver(databasename, maxIterations, arraysToSave)
solver.dataSaverHandler.set_parallel_environment(parallelEnvironment, MPIrank)

# logging ??
while solver.ok:
     Implicit time advance: iterate to solve the nonlinear equation!
     solver.TakeTimestep()
    
#print("Using pseudo-GENE, python-GENE interface code initialized OK!")



if solver.reached_end == True:
    if MPIrank==0:
        print("The solution has been reached successfully.")
        print("Took {} iterations".format(solver.l))
else:
    if MPIrank==0:
        print("The solver failed for some reason.")
        
# other logging??