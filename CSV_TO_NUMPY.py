import numpy as np
import csv

# Function to load data from CSV file and convert it to a NumPy array
def load_csv_to_numpy(csv_file):
    data = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append([float(value) for value in row])
    return np.array(data)

# Example usage:
csv_file_path = r'C:\MOUTH STUFF\output_csv_arrays\frame_20.csv'  # Replace this with the path to your CSV file
numpy_array = load_csv_to_numpy(csv_file_path)

# Save the loaded NumPy array as a .npy file in the same directory as the script
np.save('output_array.npy', numpy_array)
print("NumPy array saved as 'output_array.npy' in the same directory as the script.")
