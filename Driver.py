# Using Ocean Going Robots to Observe Wave Conditions
# Directional Visualization Framework

# Driver script

# local script imports
import waveParamters
# import spectralAnalysis
import NonDirectionalVisualization
import Logger
import Polar

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
    parser.add_argument('--csv', type=str, metavar ="data.csv", required=True, 
        help = "File name of the csv data to read")

    args = parser.parse_args()

    try:
        fp = open(args.csv, 'r')
    except FileNotFoundError:
        raise Exception("No CSV file with the given name found.") from None

    df = pd.DataFrame(pd.read_csv(fp))

    # empty brackets represent estimation method
    firstFive, spectrum = analyzeWaveData(df, "welch", "", args.sample)

    polar_plot(spectrum, 0.025, 0.56)




main()