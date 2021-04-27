# ppd_logger.py
#
# Using Ocean Going Robots to Observe Wave Conditions
# Pre-Processed Data Logger
# 
# Benjamin Lee, Mayinger
# mayingeb@oregonstate.edu
#
#
# How to call this script on the command line:
# python Logger.py your_file.csv



# imports
import numpy as np
import csv
import sys



def main():

	# print('Number of arguments:', len(sys.argv), 'arguments.')
	# print('Argument List:', str(sys.argv))
	print()

	# too many arguments are given
	if len(sys.argv) > 2:
		print("Error: too many arguments are given.")
		print("Script must be called with the following format: python Logger.py <your file>.csv")
		return

	# too few arguments are given
	if len(sys.argv) < 2:
		print("Error: too few arguments are given.")
		print("Script must be called with the following format: python Logger.py <your file>.csv") 
		return

	if sys.argv[1].find(".csv") == 0
		print("Error: CSV file required.")
		return

	file_name = sys.argv[1] # file_name is set to second argument, see script usage at the top
	n = 0 # n is our count variable, used to store the size of the data (i.e. the number of data entries)


	# check if file exists, if not print error message
	try:
		open(file_name, "r")

	except IOError:
		print("Error: the provided input file doesn't exist.")
		return


	# These are the components of the csv file:
	
	# Time
	t = []

	# Angular Acceleration (x, y, z)
	alphax = []
	alphay = []
	alphaz = []

	# Acceleration (x, y, z)
	ax = []
	ay = []
	az = []

	# Angular Velocity (x, y, z)
	omegax = []
	omegay = []
	omegaz = []

	# Velocity (x, y, z)
	vx = []
	vy = []
	vz = []

	# Displacement
	x = []
	y = []
	z = []

	# multidimensional data list is initialized containing all the components of the sample
	data = [t, alphax, alphay, alphaz, ax, ay, az, omegax, omegay, omegaz, vx, vy, vz, x, y, z]

	# this variable is used to index the columns
	data_index = 0


	# we open the csv file based on the command line argument
	with open(file_name, 'r') as file:
		reader = csv.reader(file)

		next(reader) # skip first row, which only contains variable names


		# iterate through the rows
		for row in reader:

			# checks if data is formatted properly
			if len(row) != 16:
				print("Error: provided data doesn't match the expected data format.")
				return

			# iterate through the columns
			for col in row:

				# column data is inputed into our data list
				data[data_index].append(float(col))

				#increment index
				data_index += 1

			# reset column indexing variable
			data_index = 0

			#increment sample size variable
			n += 1

		# close file
		file.close()


	# setting up the name of the log file to be the same name with a .txt extension instead of .csv
	log_file = file_name.replace('.csv', '')
	log_file = log_file + ".txt"
	# print(log_file)


	# open the log_file for writing
	with open("logs/" + log_file, 'w') as log:

		log.write("Pre-Processed Data Logger \n") # giving the file a heading
		log.write(log_file + "\n") # file name is included
		log.write("\n")
		log.write("\n")
		log.write("Sample Size: " + str(n) +"\n") # sample size is written to file
		log.write("Sample Duration: " + str(t[len(t) - 1] - t[0]) + " seconds \n") # sample duration in seconds is calculated
		log.write("Recording Frequency: " + str((n-1)/(t[len(t) - 1] - t[0])) + "hz \n") # recording frequency is calculated
		log.write("\n")
		log.write("\n")
		log.write("Minimums, Maximums and Averages: \n") # for each variable besides "time", a min, max and average is calculated and written to the file
		log.write("\n")
		log.write("alphax: \n")
		log.write("Min: " + str(min(alphax)) + "     Max: " + str(max(alphax)) + "     Avg: " + str(avg(alphax)) + "\n")
		log.write("\n")
		log.write("alphay: \n")
		log.write("Min: " + str(min(alphay)) + "     Max: " + str(max(alphay)) + "     Avg: " + str(avg(alphay)) + "\n")
		log.write("\n")
		log.write("alphaz: \n")
		log.write("Min: " + str(min(alphaz)) + "     Max: " + str(max(alphaz)) + "     Avg: " + str(avg(alphaz)) + "\n")
		log.write("\n")
		log.write("ax: \n")
		log.write("Min: " + str(min(ax)) + "     Max: " + str(max(ax)) + "     Avg: " + str(avg(ax)) + "\n")
		log.write("\n")
		log.write("ay: \n")
		log.write("Min: " + str(min(ay)) + "     Max: " + str(max(ay)) + "     Avg: " + str(avg(ay)) + "\n")
		log.write("\n")
		log.write("az: \n")
		log.write("Min: " + str(min(az)) + "     Max: " + str(max(az)) + "     Avg: " + str(avg(az)) + "\n")
		log.write("\n")
		log.write("omegax: \n")
		log.write("Min: " + str(min(omegax)) + "     Max: " + str(max(omegax)) + "     Avg: " + str(avg(omegax)) + "\n")
		log.write("\n")
		log.write("omegay: \n")
		log.write("Min: " + str(min(omegay)) + "     Max: " + str(max(omegay)) + "     Avg: " + str(avg(omegay)) + "\n")
		log.write("\n")
		log.write("omegaz: \n")
		log.write("Min: " + str(min(omegaz)) + "     Max: " + str(max(omegaz)) + "     Avg: " + str(avg(omegaz)) + "\n")
		log.write("\n")
		log.write("vx: \n")
		log.write("Min: " + str(min(vx)) + "     Max: " + str(max(vx)) + "     Avg: " + str(avg(vx)) + "\n")
		log.write("\n")
		log.write("vy: \n")
		log.write("Min: " + str(min(vy)) + "     Max: " + str(max(vy)) + "     Avg: " + str(avg(vy)) + "\n")
		log.write("\n")
		log.write("vz: \n")
		log.write("Min: " + str(min(vz)) + "     Max: " + str(max(vz)) + "     Avg: " + str(avg(vz)) + "\n")
		log.write("\n")
		log.write("x: \n")
		log.write("Min: " + str(min(x)) + "     Max: " + str(max(x)) + "     Avg: " + str(avg(x)) + "\n")
		log.write("\n")
		log.write("y: \n")
		log.write("Min: " + str(min(y)) + "     Max: " + str(max(y)) + "     Avg: " + str(avg(y)) + "\n")
		log.write("\n")
		log.write("z: \n")
		log.write("Min: " + str(min(z)) + "     Max: " + str(max(z)) + "     Avg: " + str(avg(z)) + "\n")
		log.write("\n")
		log.write("\n")
		log.write("\n")
		log.write("Outlier Calculation: \n") # outliers are calculated are presented for each variable, with the time that they appeared
		log.write("\n")
		log.write("alphax: \n")
		temp = find_outliers(alphax) # list of outliers is produced
		if len(temp) > 0: # if outlier list is non-empty
			for o in temp: # iterate through the list
				log.write("     Outlier: " + str(o) + " at time = " + str(t[alphax.index(o)]) + "\n") # write the outlier value and it's corresponding time to file
		else:
			log.write("No outliers found. \n") # if outlier list is empty write this message

		# the steps above are repeated for all the variables within the data set
		log.write("\n")
		log.write("alphay: \n")
		temp = find_outliers(alphay)
		if len(temp) > 0:
			for o in temp:
				log.write("     Outlier: " + str(o) + " at time = " + str(t[alphay.index(o)]) + "\n")
		else:
			log.write("No outliers found. \n")

		log.write("\n")
		log.write("alphaz: \n")
		temp = find_outliers(alphaz)
		if len(temp) > 0:
			for o in temp:
				log.write("     Outlier: " + str(o) + " at time = " + str(t[alphaz.index(o)]) + "\n")
		else:
			log.write("No outliers found. \n")

		log.write("\n")
		log.write("ax: \n")
		temp = find_outliers(ax)
		if len(temp) > 0:
			for o in temp:
				log.write("     Outlier: " + str(o) + " at time = " + str(t[ax.index(o)]) + "\n")
		else:
			log.write("No outliers found. \n")

		log.write("\n")
		log.write("ay: \n")
		temp = find_outliers(ay)
		if len(temp) > 0:
			for o in temp:
				log.write("     Outlier: " + str(o) + " at time = " + str(t[ay.index(o)]) + "\n")
		else:
			log.write("No outliers found. \n")

		log.write("\n")
		log.write("az: \n")
		temp = find_outliers(az)
		if len(temp) > 0:
			for o in temp:
				log.write("     Outlier: " + str(o) + " at time = " + str(t[az.index(o)]) + "\n")
		else:
			log.write("No outliers found. \n")

		log.write("\n")
		log.write("omegax: \n")
		temp = find_outliers(omegax)
		if len(temp) > 0:
			for o in temp:
				log.write("     Outlier: " + str(o) + " at time = " + str(t[omegax.index(o)]) + "\n")
		else:
			log.write("No outliers found. \n")

		log.write("\n")
		log.write("omegay: \n")
		temp = find_outliers(omegay)
		if len(temp) > 0:
			for o in temp:
				log.write("     Outlier: " + str(o) + " at time = " + str(t[omegay.index(o)]) + "\n")
		else:
			log.write("No outliers found. \n")

		log.write("\n")
		log.write("omegaz: \n")
		temp = find_outliers(omegaz)
		if len(temp) > 0:
			for o in temp:
				log.write("     Outlier: " + str(o) + " at time = " + str(t[omegaz.index(o)]) + "\n")
		else:
			log.write("No outliers found. \n")

		log.write("\n")
		log.write("vx: \n")
		temp = find_outliers(vx)
		if len(temp) > 0:
			for o in temp:
				log.write("     Outlier: " + str(o) + " at time = " + str(t[vx.index(o)]) + "\n")
		else:
			log.write("No outliers found. \n")

		log.write("\n")
		log.write("vy: \n")
		temp = find_outliers(vy)
		if len(temp) > 0:
			for o in temp:
				log.write("     Outlier: " + str(o) + " at time = " + str(t[vy.index(o)]) + "\n")
		else:
			log.write("No outliers found. \n")

		log.write("\n")
		log.write("vz: \n")
		temp = find_outliers(vz)
		if len(temp) > 0:
			for o in temp:
				log.write("     Outlier: " + str(o) + " at time = " + str(t[vz.index(o)]) + "\n")
		else:
			log.write("No outliers found. \n")

		log.write("\n")
		log.write("x: \n")
		temp = find_outliers(x)
		if len(temp) > 0:
			for o in temp:
				log.write("     Outlier: " + str(o) + " at time = " + str(t[x.index(o)]) + "\n")
		else:
			log.write("No outliers found. \n")

		log.write("\n")
		log.write("y: \n")
		temp = find_outliers(y)
		if len(temp) > 0:
			for o in temp:
				log.write("     Outlier: " + str(o) + " at time = " + str(t[y.index(o)]) + "\n")
		else:
			log.write("No outliers found. \n")

		log.write("\n")
		log.write("z: \n")
		temp = find_outliers(z)
		if len(temp) > 0:
			for o in temp:
				log.write("     Outlier: " + str(o) + " at time = " + str(t[z.index(o)]) + "\n")
		else:
			log.write("No outliers found. \n")

		# close output file
		log.close()

		# print success message
		print("Script ran successfully, a new log: " + log_file + " has been created.")

# this function computes the average of a list
def avg(list):
	return sum(list)/len(list)

# this function computes the outliers within a list and returns them in a different list
def find_outliers(list):
    
    outliers = []
    threshold = 3
    mean = np.mean(list)
    std =np.std(list)
    
    # z-score outlier calculation, where the threshold "3" defines how many standard deviations a value needs to be to be considered an outlier
    for o in list:
    	if std != 0:
	        z_score = (o - mean)/std 
	        if np.abs(z_score) > threshold:
	            outliers.append(o)

    return outliers

# call main to run the logging script
main()