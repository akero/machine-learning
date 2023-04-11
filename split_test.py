import pandas as pd

# Load the data from the CSV file
data = pd.read_csv('test_data.csv')

# Calculate the number of rows for the validation set
validation_rows = int(len(data) * 0.3)

# Split the data into validation and new test data sets
validation_data = data[:validation_rows]
new_test_data = data[validation_rows:]

# Save the new CSV files
validation_data.to_csv('validation.csv', index=False)
new_test_data.to_csv('new_test_data.csv', index=False)