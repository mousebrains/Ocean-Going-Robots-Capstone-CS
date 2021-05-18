# Using Ocean Going Robots to Observe Wave Conditions
# Directional Visualization Framework

# Driver script

# local script imports
import scipy as sp
from waveParameters import analyzeWaveData
# import spectralAnalysis
# from NonDirectionalVisualization import

from Polar import polar_plot
from NonDirectionalVisualization import nonDirectional
import sys
import argparse
import pandas as pd
import numpy as np

sampleRate = 1.28

def main():

	# INPUT VALIDATION ##########################################################

	# too many arguments are given 
	if len(sys.argv) > 2:
		print("Error: too many arguments are given.")
		print("Script must be called with the following format: python Polar.py <your data>.csv")
		return

	# too few arguments are given
	if len(sys.argv) < 2:
		print("Error: too few arguments are given.")
		print("Script must be called with the following format: python Polar.py <your data>.csv")
		return

	# check if input is CSV file
	if sys.argv[1].find(".csv") == 0:
		print("Error: CSV file required.")
		return

	# extra arguments: 'directional/nondirectional', 'sample rate', 
	# 'estimation method (MLM/Entropy)', 'debug' 

	parser = argparse.ArgumentParser()
	parser.add_argument('--csv', type=str, metavar ="data.csv", required=True, help = "File name of the csv data to read")

	args = parser.parse_args()

	try:
		fp = open(args.csv, 'r')
	except FileNotFoundError:
		raise Exception("No CSV file with the given name found.") from None

	df = pd.DataFrame(pd.read_csv(fp))

    # empty brackets represent estimation method
	firstFiveWP, spectrum, firstFive = analyzeWaveData(df, "welch", "", sampleRate)

	spectrum *= np.pi/ 180

	data = np.array(firstFive.Czz) * spectrum

	data = np.transpose(data)

	polar_plot(data, 0.025, 0.56)

	spectrum2 = firstFive.Czz
	freqBins = firstFive.freq

	
	nonDirectional(freqBins, spectrum2)

main()