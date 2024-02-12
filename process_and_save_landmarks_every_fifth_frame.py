import os
import numpy as np
import cv2
import dlib

# Initialize the face detector (Dlib) and facial landmark predictor (Dlib)
detector = dlib.get_frontal_face_detector()
p = 'shape_predictor_68_face_landmarks.dat'  # Path to Dlib's pre-trained shape predictor model
predictor = dlib.shape_predictor(p)

input_directory = 'output_numpy_arrays'
output_directory = 'LANDMARK_OUTPUT'
os.makedirs(output_directory, exist_ok=True)

# List files in the input directory
files = sorted([file for file in os.listdir(input_directory) if file.startswith('frame_') and file.endswith('.npy')])

# Process every fifth file in the list
for i in range(0, len(files), 5):
    file_name = files[i]
    file_path = os.path.join(input_directory, file_name)

    # Load normalized grayscale image from NumPy array
    normalized_image = np.load(file_path)

    # Convert normalized numpy array to 8-bit unsigned integer format (0 to 255)
    grayscale_image = (normalized_image * 255).astype(np.uint8)

    # Detect faces in the grayscale image using Dlib
    faces = detector(grayscale_image)

    # Draw facial landmarks on the image
    for face in faces:
        x, y, width, height = face.left(), face.top(), face.width(), face.height()

        # Predict facial landmarks
        shape = predictor(grayscale_image, face)
        shape = np.array([[point.x, point.y] for point in shape.parts()])

        # Draw green dots on the image for facial landmarks
        for (x, y) in shape:
            cv2.circle(grayscale_image, (x, y), 2, (0, 255, 0), -1)

    # Save the image with facial landmarks as a PNG file
    output_file = os.path.join(output_directory, f'{file_name.replace(".npy", ".png")}')
    cv2.imwrite(output_file, grayscale_image)
    print(f"Image with facial landmarks saved as {output_file}")
