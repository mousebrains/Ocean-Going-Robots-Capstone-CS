import numpy as np
import pandas as pd
import scipy as sp
import argparse
# import yaml
import scipy
import sys
import spectralAnalysis as sa
from scipy import signal
from scipy import fft

def analyzeWaveData(df:pd.DataFrame, fftmethod:str, dsfmethod:str, sampleRate:int=10):
    #sampleRate = 1/df.t[1]
    nperseg = np.floor(len(df)/8) 
    if fftmethod == "rfft":
        firstFive, spectrum = sa.displacementToRfft(df, sampleRate, nperseg) 
    elif fftmethod == "welch":
        firstFive, spectrum = sa.displacementToWelch(df, sampleRate, "boxcar", nperseg, True, "density") 

    # print(waveParameters(firstFive))
    return waveParameters(firstFive, spectrum), spectrum, firstFive


# This function calculates wave parameters
#
#   Inputs: Dataframe of wave moments and characteristic parameters
#
#   Outputs: a2, b2
# 
def waveParameters(df:pd.DataFrame, DS:list):
    pf = pd.Series(dtype=float)
    pf['binSize'] = np.mean(np.diff(df.freq))
    pf["m0"] = df.Czz.sum() * pf.binSize
    pf["m1"] = (df.Czz * df.freq).sum() * pf.binSize
    pf["m2"] = (df.Czz * np.square(df.freq)).sum() * pf.binSize
    pf["Hm0"] = 4 * np.sqrt(pf.m0)
    pf["Tav"] = pf.m0 / pf.m1
    pf["Tzero"] = np.sqrt(pf.m0 / pf.m2)
    pf["Tp"] = 1/df.freq[np.argmax(df.Czz)]
    #print(np.shape(DS))
    a1Hat = 1/pf["m0"] * scipy.integrate.simpson(df.a1 * df.Czz, dx=0.0044)
    b1Hat = 1/pf["m0"] * scipy.integrate.simpson(df.b1 * df.Czz, dx=0.0044)
    a2Hat = 1/pf["m0"] * scipy.integrate.simpson(df.a2 * df.Czz, dx=0.0044)
    b2Hat = 1/pf["m0"] * scipy.integrate.simpson(df.b2 * df.Czz, dx=0.0044)
    pf["Dmean"] = (180/np.pi)*np.arctan2(b1Hat, a1Hat)
    return pf

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, metavar ="data.csv", required=True, 
        help = "File name of the csv data to read")
    parser.add_argument('sample', type=float, nargs="?", metavar ="fs", default="10",
        help = "samplerate")
    #parser.add_argument('window', type=str, nargs="?", metavar ="wind", default="han",
    #    help = "Windowing function to apply")
    #parser.add_argument('nseg', type=int, nargs="?", metavar ="nseg", default=1200, 
    #    help = "Number of samples per segment for welch's method")
    args = parser.parse_args()

    try:
        fp = open(args.csv, 'r')
    except FileNotFoundError:
        raise Exception("No CSV file with the given name found.") from None

    df = pd.DataFrame(pd.read_csv(fp))
 
    #print(np.fft.rfft())
    #first parameter = dataframe of displacements or accelerations
    #second parameter = fft method: rfft or welch
    #third parameter = dsf estimation method
    #fourth parameter = sample rate
    ff, sp = analyzeWaveData(df, "welch", "", args.sample)
    print(waveParameters(ff, sp))








    