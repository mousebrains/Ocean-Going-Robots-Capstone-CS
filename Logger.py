# Logger.py
#
# Using Ocean Going Robots to Observe Wave Conditions
# Pre-Processed Data Logger
# 
# Benjamin Lee, Mayinger
# mayingeb@oregonstate.edu
#
#
# How to call this script on the command line:
# python Logger.py your_file.csv skipFirst
# skipFirst is a boolean value wherin:
# 0 = don't skip first row, 1 = skip first row



# imports
import numpy as np
import csv
import sys



def main():

	# print('Number of arguments:', len(sys.argv), 'arguments.')
	# print('Argument List:', str(sys.argv))
	print()

	# too many arguments are given
	if len(sys.argv) > 3:
		print("Error: too many arguments are given.")
		print("Script must be called with the following format: python Logger.py <your file>.csv")
		return

	# too few arguments are given
	if len(sys.argv) < 3:
		print("Error: too few arguments are given.")
		print("Script must be called with the following format: python Logger.py <your file>.csv") 
		return

	if sys.argv[1].find(".csv") == 0:
		print("Error: CSV file required.")
		return

	file_name = sys.argv[1] # file_name is set to second argument, see script usage at the top
	skipFirst = sys.argv[2] # skipFirst is the boolean value that determines whether the first row should be skipped


	# check if file exists, if not print error message
	try:
		open(file_name, "r")

	except IOError:
		print("Error: the provided input file doesn't exist.")
		return

	datalist = []
	n = 0 
	m = 0
	first = 0

	# we open the csv file based on the command line argument
	with open(file_name, 'r') as file:
		reader = csv.reader(file)

		# iterate through the rows
		for row in reader:

			if(skipFirst == '1' and first == 0):
				next(reader) # skip first row, which only contains variable names
				first = 1

			# checks if data is formatted properly
			#if len(row) != 16:
				#print("Error: provided data doesn't match the expected data format.")
				#return

			datalist.append(row)

			#increment row size variable
			n += 1

		# close file
		m = len(row)
		file.close()


	# setting up the name of the log file to be the same name with a .txt extension instead of .csv
	log_file = file_name.replace('.csv', '')
	log_file = log_file + ".txt"
	# print(log_file)

	data = np.array(datalist)

	#print(data.shape)


	# open the log_file for writing

	with open("logs/" + log_file, 'w') as log:

		log.write("CSV Data Logger \n") # giving the file a heading
		log.write(log_file + "\n") # file name is included
		log.write("\n")
		log.write("\n")
		log.write("Total Elements: " + str(n*m) +"\n") # sample size is written to file
		log.write("Data Shape: " + str(n) + "x" + str(m) + "\n")
		
		log.write("\n")
		log.write("\n")
		log.write("Row Minimums, Maximums and Averages: \n") # for each variable besides "time", a min, max and average is calculated and written to the file
		log.write("\n")
		
		for i in range(n):
			log.write("Row Element: " + str(i) + "		Min: " + str(min(data[i])) + "     Max: " + str(max(data[i])) + "     Avg: " + str(round(row_avg(data, i, m), 7)) + "\n")

		log.write("\n")
		log.write("\n")
		log.write("Column Minimums, Maximums and Averages: \n") # for each variable besides "time", a min, max and average is calculated and written to the file
		log.write("\n")

		for i in range(m):
			log.write("Column Element: " + str(i) + "		Min: " + str(col_min(data, i, n)) + "     Max: " + str(col_max(data, i, n)) + "     Avg: " + str(round(col_avg(data, i, n), 7)) + "\n")

		# close output file
		log.close()

		# print success message
		print("Script ran successfully, a new log: " + log_file + " has been created.")

# this function computes row averages
def row_avg(data, row, row_len):
	
	sum = 0.0

	for i in range(row_len):
		sum += float(data[row][i])

	return (sum/row_len)

# this function computes column averages
def col_avg(data, col, col_len):

	sum = 0.0

	for i in range(col_len):
		sum += float(data[i][col])

	return (sum/col_len)

# this function computes column min
def col_min(data, col, col_len):

	min = 1000000;

	for i in range(col_len):
		if(float(data[i][col]) < min):
			min = float(data[i][col])

	return min

# this function computes column max
def col_max(data, col, col_len):

	max = 0;

	for i in range(col_len):
		if(float(data[i][col]) > max):
			max = float(data[i][col])

	return max

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