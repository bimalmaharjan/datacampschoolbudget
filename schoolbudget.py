import pandas as pd 
import matplotlib.pyplot as plt

# this section will just have code, I won't execute because I couldn't download 
# TrainingData.csv

def import_file():
	df = pd.read_csv('TrainingData.csv', index_col = 0)
	return df

def eda(df):
	print df.head()
	print df.tail()
	print df.describe()
	print df.info()

		# Create the histogram
	plt.hist(df['FTE'].dropna())

	# Add title and labels
	plt.title('Distribution of %full-time \n employee works')
	plt.xlabel('% of full-time')
	plt.ylabel('num employees')

	# Display the histogram
	plt.show()

def exploring_datatypes(df):
	print df.dtypes
	print df.dtypes.value_counts()

def categorize_dataset(df):

	LABELS = ['Function', 'Use', 'Sharing', 'Reporting', 'Student_Type', 
			'Position_Type', 'Object_Type', 'Pre_K', 'Operating_Status']
	# Define the lambda function: categorize_label
	categorize_label = lambda x: x.astype('category')

	# Convert df[LABELS] to a categorical type
	df[LABELS] = df[LABELS].apply(categorize_label, axis =0)

	# Print the converted dtypes
	print(df[LABELS].dtypes)

		# Calculate number of unique values for each label: num_unique_labels
	num_unique_labels = df[LABELS].apply(pd.Series.nunique)

	# Plot number of unique values for each label
	num_unique_labels.plot(kind='bar')

	# Label the axes
	plt.xlabel('Labels')
	plt.ylabel('Number of unique values')

	# Display the plot
	plt.show()

	return df








if __name__ == '__main__':
	
	df = import_file()
	eda(df)
	exploring_datatypes(df)
	# change the object data type to categorize data type
	# so that it is easy to convert to float data type

	df = categorize_dataset(df)

