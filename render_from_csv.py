import csv
import numpy as np
import cv2

def read_csv_file(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = [list(map(float, row)) for row in reader]
        return np.array(data)

def reconstruct_image(data_array):
    height, width = data_array.shape
    reconstructed_image = (data_array * 255).astype(np.uint8)
    reconstructed_image = cv2.resize(reconstructed_image, (width, height))
    return reconstructed_image

def save_image(image, output_path):
    cv2.imwrite(output_path, image)

# Example usage
csv_file_path = r"C:\MOUTH STUFF\output_csv_arrays\frame_0.csv"
output_image_path = 'reconstructed_image.png'

data_array = read_csv_file(csv_file_path)
reconstructed_image = reconstruct_image(data_array)
save_image(reconstructed_image, output_image_path)
