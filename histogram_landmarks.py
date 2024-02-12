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

# Process every file in the list
for file_name in files:
    file_path = os.path.join(input_directory, file_name)

    # Load normalized grayscale image from NumPy array
    normalized_image = np.load(file_path)

    # Convert normalized numpy array to 8-bit unsigned integer format (0 to 255)
    grayscale_image = (normalized_image * 255).astype(np.uint8)

    # Apply histogram equalization to enhance contrast
    equalized_image = cv2.equalizeHist(grayscale_image)

    # Detect faces in the equalized grayscale image using Dlib
    faces = detector(equalized_image)

    # Draw facial landmarks on the image if faces are detected
    if len(faces) > 0:
        for face in faces:
            x, y, width, height = face.left(), face.top(), face.width(), face.height()

            # Predict facial landmarks
            shape = predictor(equalized_image, face)
            shape = np.array([[point.x, point.y] for point in shape.parts()])

            # Draw white dots on the image for facial landmarks
            for (x, y) in shape:
                cv2.circle(equalized_image, (x, y), 2, (255, 255, 255), -1)

        # Save the image with facial landmarks as a PNG file
        output_file = os.path.join(output_directory, f'{file_name.replace(".npy", ".png")}')
        cv2.imwrite(output_file, equalized_image)
        print(f"Image with facial landmarks saved as {output_file}")
    else:
        print(f"No faces detected in {file_name}.")
