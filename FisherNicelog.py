import utility_functions as util
import MCMC_nicelog as nicelog
import numpy as np
from copy import deepcopy
from timeit import default_timer as timer

'''
This program will do Fisher analysis for any function of one variable,
with N parameters. As inputs are needed the function, the datapoints and the
assumed values of the parameters of the input function. These are given in
steps 1-3.

In steps 4 we calculate the needed derivatives for the Fisher sum. We need one
per datapoint per parameter, so the total number of derivatives is the product
of the number of input parameters and the number of datapoints.

1) First we define the function we'd like to analyse
2) Then we input the datapoints and their uncertainties
3) Then we define the assumed parameters of the input function

4) Calculate the derivatives needed for Fisher sum
5) Calculate Fisher sum by doing (N'*(C**2))*N where N is the numerical derivative matrix and
C is the inverse of the diagonal covariance matrix.  Or alternatively just sum over all the
individual components of the FIsher matrix. 
'''

start = timer() #To track program runtime

#---------------------------------- INPUTS -----------------------------------
#1) The function that we wish to take derivatives of
def inputFunction(dataPoints,baseParameters):

    return nicelog.nicelog_snr(np.log10(dataPoints),baseParameters)

#2) The datapoints where we wish to know the derivative
ssfr, snr, snr_err = util.read_data()
dataPoints = ssfr
dataPointsUncertainties = snr_err

#3) The parameter values of the function


#baseParameters = np.array([5.222222e-14, 0.2272727, 1.30909e-10, 0.9449494],dtype=np.float64)
#baseParametersOriginal = np.array([5.222222e-14, 0.2272727, 1.309090e-10, 0.9449494],dtype=np.float64)

baseParameters = np.array([1.1183673469387756e-13, 0.49081632653061219, 1.6653061224489797e-10, 0.7265306122448979],dtype=np.float64)
baseParametersOriginal = np.array([1.1183673469387756e-13, 0.49081632653061219, 1.6653061224489797e-10, 0.7265306122448979],dtype=np.float64)

#----------------------------- FUNCTION DEFINITIONS---------------------------
def derivativeNumerical(function,stepSize,parameterNumber):
    
    #Check if the stepSize is below computational limit, increasing if is
    if (stepSize < np.sqrt(np.spacing(1))):
         stepSize = np.sqrt(np.spacing(1))
    
    #Taking deepcopy of original baseParameters to reset baseParameters
    baseParameters = deepcopy(baseParametersOriginal)
    
    #Determining the appropriate stepSize for the parameter in question
    stepSize_temp = baseParameters[parameterNumber] + stepSize
    stepSize = stepSize_temp - baseParameters[parameterNumber]
    
    #Calculating the right and left terms 
    baseParameters[parameterNumber] = baseParametersOriginal[parameterNumber] + stepSize
    rightStep = function(dataPoints,baseParameters)
    
    baseParameters[parameterNumber] = baseParametersOriginal[parameterNumber] - stepSize
    leftStep = function(dataPoints,baseParameters)
    
    return (rightStep-leftStep) / (2.*stepSize)

def numericalDerivativeParameters(function,baseParameters,dataPoints):
    Numderiv = np.zeros((len(dataPoints),len(baseParameters)))
    
    stepSize = 1e3 * np.spacing(1)
    for i in np.arange(len(baseParameters)):
        stepSizeTemp = baseParameters[i] - stepSize
        stepSize = stepSizeTemp - baseParameters[i]
        print stepSize
        Numderiv[:,i] = derivativeNumerical(function,stepSize,i)

    return Numderiv
#---------------------------------- MAIN ------------------------------------

#4) Defining numerical derivatives
Numderiv = np.zeros((len(dataPoints),len(baseParameters)),dtype=np.float128)
Numderiv = numericalDerivativeParameters(inputFunction,baseParameters,dataPoints)

#5)

Fisher = np.zeros((len(baseParameters),len(baseParameters)))

'''
for i in range(len(baseParameters)):
    for j in range(i,len(baseParameters)):
        Fisher[i,j] = np.sum(Numderiv[:,i]*Numderiv[:,j] / (dataPointsUncertainties**2))
        Fisher[j,i] = Fisher[i,j]

print "Fisher matrix: "
print Fisher
'''


Covar = np.diag(dataPointsUncertainties)
Covar_inv = np.linalg.inv(Covar)

Fisher = np.dot(np.dot(Numderiv.T,Covar_inv**2),Numderiv)

print Fisher

Covar = np.linalg.inv(Fisher)
print "\nCovariance matrix:"
print Covar
print np.sqrt(Covar)

#print np.dot(Fisher,np.linalg.inv(Fisher)) #To check if the covariance matrix is calculated correctly
print "In time: ", timer() - start