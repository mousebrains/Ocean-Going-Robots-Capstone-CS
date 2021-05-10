import numpy as np
import pandas as pd
import scipy as sp
import argparse
import scipy
import sys
from scipy import signal
from scipy import fft

#This function calculates PSD's and Fourier Coeffecients for Wave Spectra using numpys rfft function
#
#   Inputs: Pandas.DataFrame, sampling rate = fs, segment length (welch) = segLength, one vs two-sided = sided, fft scale (density vs spectrum)
#
#   Outputs: Pandas.DataFrame
#
# todo: convert from accel first, figure out format after export from ECE team
def displacementToRfft(df:pd.DataFrame, fs:int, nseg:int) -> pd.DataFrame:

    #Calculate displacement series
    
    xSeries = 1 * mkSeries(df, "x")
    ySeries = -1 * mkSeries(df, "y")
    zSeries = mkSeries(df, "z")
    dt = 1/fs
    
    rf = pd.DataFrame({'freq':np.fft.rfftfreq(len(zSeries), dt)}) 
    N = len(rf.freq) -1
    
    #Calculate psd

    rf["X"] = dt * np.fft.rfft(xSeries) # FFT of x position, m/Hz
    rf["Y"] = dt * np.fft.rfft(ySeries) # FFT of y position, m/Hz
    rf["Z"] = dt * np.fft.rfft(zSeries) # FFT of z position, m/Hz

    #Calculate the co and quadrature spectra
    #discard the frequency, values are the same for our given fs and nperseg
    rf['Cxx'] = np.real(rf.X * np.conjugate(rf.X))  / (N * dt)
    rf['Cyy'] = np.real(rf.Y * np.conjugate(rf.Y))  / (N * dt)
    rf['Czz'] = np.real(rf.Z * np.conjugate(rf.Z))  / (N * dt)

    rf["Qzx"] = np.imag(rf.Z * np.conjugate(rf.X)) / (N) # m^2/Hz
    rf["Qzy"] = np.imag(rf.Z * np.conjugate(rf.Y)) / (N) # m^2/Hz
    rf["Cxy"] = np.real(rf.X * np.conjugate(rf.Y)) / (N) # m^2/Hz


    #Calculate Fourier Components
    
    rf['a0'] = zeroOrderFourier(rf['Czz'])
    rf['a1'], rf['b1'] = firstOrderFourier(rf['Qzx'], rf['Cxx'], rf['Cyy'], rf['Czz'], rf['Qzy'])
    rf['a2'], rf['b2'] = secondOrderFourier(rf['Cxx'], rf['Cyy'], rf['Cyy'], rf['Cxy'])

    
    #only resolve freq greater than 0 and less than .6
    rf = rf[rf.freq > 0]

    #rf = rf[rf.freq < .6]
    d = 571
    #create freq and direction matricies for linalg
    freq = rf.freq.to_numpy()
    theta = np.radians(np.arange(0, 360+1, 5))
    (freqG, thetaG) = np.meshgrid(freq, theta)
    
    #inverse oof the elements of the cross spectral matrix
    Ds = Gmnx(rf.Z, rf.X, rf.Y, dt, N)

    #wave number from dispersion relation
    rf['k'] = waveNum(rf.freq)
    #rf['k'] = np.sqrt((rf['Cxx'] + rf['Cyy'])/rf['Czz'])
    k = rf.k.to_numpy()

    mlm = mlmEstimate(Ds, thetaG, k, d)


    kFactor = .2 #mlm estimate is given as kappa/mlm. not sure how to calculate kappa, but it should be around .002. no effect on direction.
    # !!!! PLOT HERE !!!!!!!!
    
    #Polar.polar_plot(rf, 0, 1/np.array(mlm)) 
    #Polar.polar_plot(rf, 0, 1/np.array(mlm))

    return rf, mlm
    
#This function calculates PSD's and Fourier Coeffecients for Wave Spectra using numpys welch method
#
#   Inputs: Pandas.DataFrame, sampling rate = fs, segment length (welch) = segLength, one vs two-sided = sided, fft scale (density vs spectrum)
#
#   Outputs: Pandas.DataFrame
#
def displacementToWelch(df:pd.DataFrame, fs:int, wind:list, segLength:int, sided:bool, scale:list) -> pd.DataFrame:

    #calculate mission duration in seconds
    #missionDuration = df['t'].max()
    #Calculate displacement series
    
    xSeries = 1 * mkSeries(df, "x")
    ySeries = -1 * mkSeries(df, "y")
    zSeries = mkSeries(df, "z")
    dt = 1/fs
    
    #Calculate psd 
    Ds = seriesToCrossSpectrum(zSeries, xSeries, ySeries, fs, wind, segLength, sided, scale)
    Gmn = zxyMatrixInverse(Ds)

    xF, _ = signal.welch(xSeries, fs, window=wind, nperseg=segLength, return_onesided=sided, scaling=scale, detrend=False)
    # yF, yP = signal.welch(ySeries, fs, window=wind, nperseg=segLength, return_onesided=sided, scaling=scale, detrend=False)
    # zF, zP = signal.welch(zSeries, fs, window=wind, nperseg=segLength, return_onesided=sided, scaling=scale, detrend=False)
    rf = pd.DataFrame({'freq':xF[1:]})

    
    # rf['Cxx'] = np.real(xP)
    # rf['Cyy'] = np.real(yP)
    # rf['Czz'] = np.real(zP)
    rf['Cxx'] = np.real(Ds[1][1])
    rf['Cyy'] = np.real(Ds[2][2])
    rf['Czz'] = np.real(Ds[0][0])
    rf['Cxy'] = np.real(Ds[1][2])
    rf['Qzx'] = np.imag(Ds[0][1])
    rf['Qzy'] = np.imag(Ds[0][2])

    # #Calculate the co and quadrature spectra
    # #discard the frequency, values are the same for our given fs and nperseg
    
    # rf['Cxy'] = np.real(signal.csd(xSeries, ySeries, fs, window=wind, nperseg=segLength, return_onesided=sided,  scaling=scale)[1])
    # rf['Qzx'] = np.imag(signal.csd(zSeries, xSeries, fs, window=wind, nperseg=segLength, return_onesided=sided,  scaling=scale)[1])
    # rf['Qzy'] = np.imag(signal.csd(zSeries, ySeries, fs, window=wind, nperseg=segLength, return_onesided=sided,  scaling=scale)[1])


    #Calculate Fourier Components
    
    # rf['a0'] = zeroOrderFourier(rf['Czz'])
    # rf['a1'], rf['b1'] = firstOrderFourier(rf['Qzx'], rf['Cxx'], rf['Cyy'], rf['Czz'], rf['Qzy'])
    # rf['a2'], rf['b2'] = secondOrderFourier(rf['Cxx'], rf['Cyy'], rf['Cyy'], rf['Cxy'])

    
    #only resolve freq greater than 0 and less than .6
    rf = rf[rf.freq > 0]
    #rf = rf[rf.freq < .6]
    d = 571

    #create freq and direction matricies for linalg
    freq = rf.freq.to_numpy()
    theta = np.radians(np.arange(0, 360+1, 5))
    (freqG, thetaG) = np.meshgrid(freq, theta)
    #print(np.shape(theta))
    #wave number from dispersion relation
    rf['k'] = waveNum(rf.freq)
    #rf['k'] = np.sqrt((rf['Cxx'] + rf['Cyy'])/rf['Czz'])
    k = rf.k.to_numpy()

    mlm = mlmEstimate(Gmn, thetaG, k, d)
    #mem = memEstimate(rf, thetaG, k, d)
    
    #gradient descent convergence variables
    beta =  5
    gamma = 50

    #curr = imlmEstimate(mlm, rf.Czz, freq, theta, d, beta, gamma)
    newst = spectrumToCrossSpectrum(mlm, k, theta, d, freq)
    Cxx = np.real(newst[1][1])
    Cyy = np.real(newst[2][2])
    Czz = np.real(newst[0][0])
    Cxy = np.real(newst[1][2])
    Qzx = np.imag(newst[0][1])
    Qzy = np.imag(newst[0][2])


    rf['a0'] = newst[0][0]
    rf['a1'], rf['b1'] = firstOrderFourier(Qzx, Cxx, Cyy, Czz, Qzy)
    rf['a2'], rf['b2'] = secondOrderFourier(Cxx, Cyy, Cyy, Cxy)

    kFactor = .2 #mlm estimate is given as kappa/mlm. not sure how to calculate kappa, but it should be around .002. no effect on direction.
    # !!!! PLOT HERE !!!!!!!!

    #MyPolar.plotit(rf, 0, np.array(mlm))
    #estim = 1/np.array(mlm) 
    #Polar.polar_plot(estim * (np.pi/180) * E, 6000, 0)
    
    #I've commented out MyPolar here and in the header, its the script I've used for visualization
    #needs to be in the same directory, then should work

    return rf, mlm

def mlmEstimate(Ds:np.array, thetaG:list, k:int, d:float):
    #inverse oof the elements of the cross spectral matrix
    #Ds = Gmn(zSeries, xSeries, ySeries, fs, wind, segLength, sided, scale)
    #Transfer functions given by benoit
    numerator = k * (d + .5)
    denom = k * d
    if(k.any() * d > 15):
        hxy = 1j * k**0
        hz = k**0 
    else:
        hxy = 1j* np.cosh(numerator)/np.sinh(denom)
        hz = np.sinh(numerator)/np.sinh(denom)  
    hmatrix = [hz, hxy, -hxy] # matrix operations
    #hmatrix = [1, 1, 1]
    
    #alpha = lambda f : [(1, 1), (1j * np.cos(f), 1), (1, 1j * np.sin(f))]
    alpha = lambda f : [1, np.cos(f), -np.sin(f)]
    mlmsum = 0
    mlm = []
    #print(np.shape(thetaG))
    for f in thetaG: #from benoit, not sure how to translate the conditions for alpha and beta into matrix operations...
        mlmsum = 0
        for m in range(0, 3):
            for n in range(0, 3):
                mlmsum += np.real((hmatrix[m] * alpha(f)[m]) *  Ds[m][n] * np.conj(alpha(f)[n] * hmatrix[n]))
        mlm.append(mlmsum)

    return 1/np.array(mlm)

def memEstimate(firstFive:pd.DataFrame, thetaG:list, k:float, d:float):
    i = 1j
    c1 = firstFive.a1 + (i * firstFive.b1)
    c2 = firstFive.a2 + (i * firstFive.b2)
    c1Conjugate = np.transpose(c1)
    c2Conjugate = np.transpose(c2)
    F1 = (c1 - c2*c1Conjugate)/(1-np.abs(c1)**2)
    F2 = c2 - (c1 * F1)
    mem = []
    for delta in thetaG:
        numerator = 1 - (F1 * c1Conjugate) - (F2 * c2Conjugate)
        denom = np.abs(1 - F1*(np.cos(delta) - i * np.sin(delta)) - F2 * (np.cos(2*delta) - i * np.sin(2*delta)))**2
        mem.append(1/(2*np.pi) * (numerator/denom))
    return mem

def bettermem():
    alpha0 = np.full(73, 1)
    alpha1 = np.cos(theta)
    alpha2 = np.sin(theta)
    alpha3 = np.cos(2*theta)
    alpha4 = np.sin(2*theta) 
    beta1 = a1/a0
    beta2 = b1/a0
    beta3 = a2/a0
    beta4 = b2/a0
    alphaMatrix = [alpha0, alpha1, alpha2, alpha3, alpha4]
    betaMatrix = [beta1, beta2, beta3, beta4]
    for i in range(4):
        integrandsum = 1 
        integrand = (betaMatrix[i] - alphaMatrix[i+1]) @ np.exp()


def imlmEstimate(initialEstimate:list, initialCzz:list, freq:list, theta:list, d:float, betaHP:int, gammaHP:int):
    k = waveNum(freq)
    (freqG, thetaG) = np.meshgrid(freq, theta)
    E = np.array(initialCzz)
    prev = 0.0
    curr = initialEstimate
    for i in range(10): #iterative mlm from benoit (1984)
        lamb = initialEstimate - prev
        eFactor = (np.abs(lamb)**(betaHP + 1.0))/(lamb * gammaHP)
        #curr = prev + eFactor #adjust the mlm estimate 
        crossEstimate = spectrumToCrossSpectrum(E/np.array(lamb), k, theta, d, freq) #calculate the cross spectra of the new estimate
        E = crossEstimate[0][0]
        #prev = curr
        curr = mlmEstimate(crossEstimate, thetaG, k, d) + eFactor #calculate a new estimate using the estimated spectra
    return curr
    

#Calculate the cross spectrum from a directional spectrum: used for iterative MLM
def spectrumToCrossSpectrum(S:np.array, k:np.array, theta:np.array, d:float, freq:np.array):

    numerator = k * (d + .5)
    denom = k * d
    if(k.any() * d > 15):
        hxy = 1j * k**0
        hz = k**0 
    else:
        hxy = 1j* np.cosh(numerator)/np.sinh(denom)
        hz = np.sinh(numerator)/np.sinh(denom)  
    hmatrix = [hz, hxy, -hxy] # matrix operations

    Gzz = np.trapz((hmatrix[0] * np.conjugate(hmatrix[0]) * S), x=theta, axis=0)
    Gzx = np.trapz((hmatrix[0] * np.conjugate(hmatrix[1]) * S), x=theta, axis=0)
    Gzy = np.trapz((hmatrix[0] * np.conjugate(hmatrix[2]) * S), x=theta, axis=0)

    Gxz = np.trapz((hmatrix[1] * np.conjugate(hmatrix[0]) * S), x=theta, axis=0)
    Gxx = np.trapz((hmatrix[1] * np.conjugate(hmatrix[1]) * S), x=theta, axis=0)
    Gxy = np.trapz((hmatrix[1] * np.conjugate(hmatrix[2]) * S), x=theta, axis=0)

    Gyz = np.trapz((hmatrix[2] * np.conjugate(hmatrix[0]) * S), x=theta, axis=0)
    Gyx = np.trapz((hmatrix[2] * np.conjugate(hmatrix[1]) * S), x=theta, axis=0)
    Gyy = np.trapz((hmatrix[2] * np.conjugate(hmatrix[2]) * S), x=theta, axis=0)

    G = np.array([[Gzz, Gzx, Gzy], [Gxz, Gxx, Gxy], [Gyz, Gyx, Gyy]])
 
    return G #np.linalg.pinv(G)

# This function calculates the inverse cross spectral matrix (Benoit)
#
#   Inputs: z, y, x time series, desired window, length of welches method segments, one or two sided, "density" or "spectrum" scale
#
#   Outputs: H
# 
def seriesToCrossSpectrum(z:list, x:list, y:list, fs:int, wind:list, segLength:int, sided:bool, scale:list) -> list:
 
    #G = [signal.csd(, z, fs, window=wind, nperseg=segLength, return_onesided=sided,  scaling=scale)[1] for i in]
    Gzz = trim(signal.csd(z, z, fs, window=wind, nperseg=segLength, return_onesided=sided,  scaling=scale)[1])
    Gzx = trim(signal.csd(z, x, fs, window=wind, nperseg=segLength, return_onesided=sided,  scaling=scale)[1])
    Gzy = trim(signal.csd(z, y, fs, window=wind, nperseg=segLength, return_onesided=sided,  scaling=scale)[1])
    #print(np.shape(Gzz))

    Gxz = trim(signal.csd(x, z, fs, window=wind, nperseg=segLength, return_onesided=sided,  scaling=scale)[1])
    Gxx = trim(signal.csd(x, x, fs, window=wind, nperseg=segLength, return_onesided=sided,  scaling=scale)[1])
    Gxy = trim(signal.csd(x, y, fs, window=wind, nperseg=segLength, return_onesided=sided,  scaling=scale)[1])

    Gyz = trim(signal.csd(y, z, fs, window=wind, nperseg=segLength, return_onesided=sided,  scaling=scale)[1])
    Gyx = trim(signal.csd(y, x, fs, window=wind, nperseg=segLength, return_onesided=sided,  scaling=scale)[1])
    Gyy = trim(signal.csd(y, y, fs, window=wind, nperseg=segLength, return_onesided=sided,  scaling=scale)[1])

    G = np.array([[Gzz, Gzx, Gzy], [Gxz, Gxx, Gxy], [Gyz, Gyx, Gyy]])
 
    return G #np.linalg.pinv(G)

def Gmnx(z:list, x:list, y:list, dt:int, N:int) -> list:

    #G = [signal.csd(, z, fs, window=wind, nperseg=segLength, return_onesided=sided,  scaling=scale)[1] for i in]
    Gzz = np.real(z * np.conjugate(z)  / (N * dt))
    Gzx = np.imag(z * np.conjugate(x)  / (N * dt))
    Gzy = np.imag(z * np.conjugate(y)  / (N * dt))

    Gxz = np.imag(x * np.conjugate(z)  / (N * dt))
    Gxx = np.real(x * np.conjugate(x)  / (N * dt))
    Gxy = np.real(x * np.conjugate(y)  / (N * dt))

    Gyz = np.imag(y * np.conjugate(z)  / (N * dt))
    Gyx = np.imag(y * np.conjugate(x)  / (N * dt))
    Gyy = np.real(y * np.conjugate(y)  / (N * dt))

    G = np.array([[Gzz, Gzx, Gzy], [Gxz, Gxx, Gxy], [Gyz, Gyx, Gyy]])
    #print(np.shape(G))
    return G #np.linalg.pinv(G)

def zxyMatrixInverse(G:list):
    Gv = np.zeros_like(G)
    for i in range(G.shape[-1]):
        Gv[:,:,i] = np.linalg.pinv(G[:,:,i])
    return Gv

#This function computes a series from a DataFrame.
#
#   Inputs: Pandas.DataFrame with fields time and "target". 
#
#   Outputs: Pandas.DataFrame of both fields.
#todo: convert accelerations into displacment
def mkSeries(df:pd.DataFrame, target:str) -> list:
    series = np.array(df[target])
    return series
    
#This function calculates a0, the first fourier coeff.
#
#   Inputs: Surface elevation frequency spectrum
#
#   Outputs: a0
# 
def zeroOrderFourier(spectrum:list) -> list:
    return spectrum

#This function calculates a1, b1, the second set of fourier coeffs.
#
#   Inputs: 2 co/quad spectrums and a list of respective wave numbers. Must all be the same size.
#
#   Outputs: a1, b1
# 
def firstOrderFourier(zx:list, xx:list, yy:list, zz:list, zy:list) -> list:
    a1 = 1 * zy/np.sqrt((xx + yy) * zz)
    b1 = 1 * zx/np.sqrt((xx + yy) * zz)
    return a1, b1

#(NOAA) This function calculates a2, b2, the first fourier coeff.
#
#   Inputs: 3 co/quad spectrums and a list of respective wave numbers. Must all be the same size.
#
#   Outputs: a2, b2
# 
def secondOrderFourier(xx:list, yy:list, zz:list, xy:list) -> list:
    a2 = (yy-xx)/np.sqrt(zz*(xx + yy))
    b2 = (2 * xy)/np.sqrt(zz*(xx + yy))
    return a2, b2

#This function estimates the wavenumber from the frequency using the deep water dispersion relation:
#https://en.wikipedia.org/wiki/Dispersion_(water_waves)#Dispersion_relation -> see the third column for "dispersion relation"
#
#   Inputs: list of frequencies. sorted or unsorted.
#
#   Outputs: list of wave numbers. each wave number corresponds to the freq at the same index, ie k[1] is the wave number for freq[1]
# 
def waveNum(freq:list) -> list:
    #todo: find numpy/scikit gravity constant
    wL = (2 * np.pi * freq)**2/9.86
    return np.array(wL)

# This function trims negative and the zero frequencies
#
#   Inputs: t spectrum
#
#   Outputs: t spectrum
# 
def trim(t:list) -> list:
    t = t[1:]
    return t